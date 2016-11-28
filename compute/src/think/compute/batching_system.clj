(ns think.compute.batching-system
  (:require [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.datatype.core :as dtype]
            [cortex.dataset :as ds]
            [clojure.set :as c-set]
            [think.parallel.core :as parallel])
  (:import [java.util ArrayDeque]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(def batch-types
  [:training :cross-validation :holdout :all])


(defprotocol PBatchingSystem
  ;;return train-config
  ;;Overall setup, called once.
  (setup [bs])
  ;;Returns a sequence where each item of the sequence contains:
  ;;{:input-buffers - vector of buffers used for input
  ;; :output-buffers - vector of buffers used for output
  ;;}
  ;;There is an option to skip the upload steps to the output buffers which
  ;;aren't necessary if you aren't doing gradient descent (e.g. any inference).
  (get-batches [bs batch-type upload-output-buffers?])
  ;;Get the output on the cpu
  (get-cpu-labels [bs batch-type])
  ;;Get the dataset underling this batch system.
  (get-dataset [bs]))



(defrecord DatasetBatchingSystem [input-names output-names ^long batch-size
                                  dataset driver stream datatype
                                  ^long batch-buffer-count])


(defn create-dataset-batching-system
  "Create the batching system to feed a dataset into a compute device.
batch buffer count stands for the number of buffer sets to use; 2 would be
double buffered batching, 3 would be triple buffer, etc."
  [input-names output-names batch-size
   dataset driver stream datatype
   & {:keys [batch-buffer-count]
      ;;Enable double buffering by default
      :or {batch-buffer-count 2}}]
  (let [shapes (ds/shapes dataset)
        invalid-labels (vec (remove (set (keys shapes))
                                    (distinct (concat input-names output-names))))]
    (when-not (= 0 (count invalid-labels))
      (throw (Exception. (format "Dataset is missing entry names: %s" invalid-labels))))
    (->DatasetBatchingSystem input-names output-names batch-size
                             dataset driver stream datatype
                             (long batch-buffer-count))))


(defn dataset-shape->array
  [^DatasetBatchingSystem batching-system shape]
  (let [driver (.driver batching-system)
        stream (.stream batching-system)
        datatype (.datatype batching-system)
        batch-size (.batch-size batching-system)]
   (if (number? shape)
     (math/new-array driver stream datatype [shape] batch-size)
     (let [{:keys [channel-count height width layout]} shape]
       (when-not (= layout ds/planar-image-layout)
         (throw (Exception. "Only planar image formats are supported at this time")))
       (math/new-array driver stream datatype batch-size channel-count height width)))))


(defn create-batch-buffer
  "Allocate a host buffer to load data to the array for things that are repeatedly loaded."
  [dev ary]
  {:device-array ary
   :host-buffer (drv/allocate-host-buffer dev (math/ecount ary) (dtype/get-datatype ary))})


(defn create-batch-buffers
  [^DatasetBatchingSystem batching-system names buffer-stream]
  (let [dataset (get-dataset batching-system)
        shapes (ds/shapes dataset)
        device (.driver batching-system)
        name-map (map #(vector
                        %
                        (create-batch-buffer device (dataset-shape->array batching-system
                                                                          (get shapes %))))
                      names)]
    {:batch-upload-stream buffer-stream
     :buffer-map (into {} name-map)}))

(defn copy-batch-data->host!
  [batch-data-seq {:keys [host-buffer] :as batch-buffer}]
  (dtype/copy-raw->item! batch-data-seq host-buffer 0)
  batch-buffer)


(defn copy-batch-host->device!
  [{:keys [device-array host-buffer]} stream]
  (drv/copy-host->device stream
                         host-buffer 0
                         (math/device-buffer device-array) 0
                         (math/ecount device-array))
  device-array)


(defn upload-batch-data
  [batch-data-seq batch-buffer stream]
  (-> (copy-batch-data->host! batch-data-seq batch-buffer)
      (copy-batch-host->device! stream)))


(extend-type DatasetBatchingSystem
  PBatchingSystem
  (setup [bs]
    (assoc bs
           :buffer-maps (vec
                         (repeatedly (.batch-buffer-count bs)
                                     #(create-batch-buffers
                                       bs
                                       (distinct (concat (.input-names bs)
                                                         (.output-names bs)))
                                       (drv/create-stream
                                        (.driver bs)))))))

  (get-batches [bs batch-type upload-output-buffers?]
    ;;Generate all the batches we are going to use.
    (let [dataset (.dataset bs)
          batch-size (.batch-size bs)
          buffer-maps (:buffer-maps bs)
          names (distinct (if upload-output-buffers?
                            (keys (first buffer-maps))
                            (.input-names bs)))
          index->names (into {} (map-indexed vector names))
          name->indexes (c-set/map-invert index->names)
          stream (.stream bs)
          output-names (if upload-output-buffers?
                         (.output-names bs)
                         [])
          buffer-count (long (max 1 (.batch-buffer-count bs)))]
      ;;Use a buffered seq so that we allow the uploads to happen ahead of time
      ;;without switching the cpu threads.  Cuda is especially sensitive to
      ;;switching of threads so we want to avoid it if at all possible.
      ;;So it may not be clear but the use of laziness is extremely precise here.
      ;;We have N host buffers and N device buffers.  We want to parallelize coalescing
      ;;the batch into the host buffers and we want to upload the host buffers into
      ;;the device buffers for then next batch while we are processing the current
      ;;batch thus providing overlap of computing and memory copying.
      ;;
      ;;We know (from the documentation) that with a queued-pmap operation we can
      ;;expect at most queue-depth+1 items in flight thus if we have n buffers we need to have
      ;;a queue depth of n-1. So we schedule coalescing the batch data into an upload
      ;;buffer using an offline thread pool allowing up to n items in flight
      ;;(by using a depth of n-1).  We can do as many batches as we have memory for
      ;;this way.
      ;;
      ;;We use a buffered seq *after* the host->device operation so that while we are
      ;;processing batch x we are uploading batch x+1.  To do this we use a buffered-seq
      ;;with a buffer-size maximum of 1 (and potentially 0 if we aren't using more than 1 buffer).
      ;;
      ;;Finally we place a synchronization event into the main compute stream so that we know
      ;;before we start operating on the data the upload has finished from the upload stream.
      (->> (ds/get-batches dataset batch-size batch-type names)
           (parallel/queued-pmap (- buffer-count 1)
                                 (fn [{:keys [batch-upload-stream buffer-map]} batch-data]
                                   (let [buffers (mapv buffer-map names)]
                                     [batch-upload-stream
                                      (mapv (fn [batch-datum buffer]
                                              (copy-batch-data->host! batch-datum
                                                                      buffer))
                                            batch-data
                                            buffers)]))
                                 (mapcat identity (repeat buffer-maps)))
           (map (fn [[stream buffers]]
                  [stream (mapv #(copy-batch-host->device! % stream)
                                buffers)]))
           ;;Try to initiate parallel computation + upload
           ;;by realizing n items of the lazy sequence.
           (parallel/buffered-seq (min 1 (- buffer-count 1)))
           (map (fn [[batch-upload-stream device-buffers]]
                  (drv/sync-event stream (drv/create-event batch-upload-stream))
                  {:input-buffers (mapv (comp device-buffers name->indexes) (.input-names bs))
                   :output-buffers (mapv (comp device-buffers name->indexes) output-names)})))))

  (get-dataset [bs] (.dataset bs))
  (get-cpu-labels [bs batch-type]
    (let [dataset (.dataset bs)
          names (distinct (.output-names bs))
          name->indexes (into {} (map-indexed (comp vec reverse vector) names))
          batch-size (.batch-size bs)
          batches (ds/get-batches dataset batch-size batch-type names)]
      (when (seq batches)
        (map (fn [idx]
               (mapcat #(nth % idx) batches))
             (map name->indexes (.output-names bs)))))))
