(ns mnist-classification.observe-networks
  (:require [clojure.java.io :as io]
            [cortex.datasets.mnist :as mnist]
            [think.image.data-augmentation :as image-aug]
            [cortex.nn.layers :as layers]
            [tfevent-sink.event-io :as eio]
            [clojure.core.matrix.stats :refer [mean]]
            [confuse.binary-class-metrics :as bcm :refer [accuracy precision
                                                          recall]]
            [confuse.multi-class-metrics :as mcm :refer [micro-avg-fmeasure
                                                         micro-avg-precision]]
            [clojure.core.matrix :as m]
            [cortex.experiment.classification :as classification]
            [cortex.experiment.train :as train]
            [cortex.util :as util]
            [cortex.experiment.util :as experiment-util]
            [mnist-classification.core :as mcc :refer [ensure-images-on-disk!
                                                       dataset-folder
                                                       image-size
                                                       initial-description
                                                       num-classes
                                                       class-mapping]])
  (:import [java.io File]))
(defn simple-net 
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/dropout 0.9)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn train-seq
  ([] (train-seq experiment-util/infinite-class-balanced-dataset))
  ([dataset-seq-fn]
   (let [training-folder (str dataset-folder "training")]
     (-> training-folder
         (experiment-util/create-dataset-from-folder class-mapping)
         (dataset-seq-fn)))))

(defn test-seq
  []
  (let [test-folder (str dataset-folder "test")]
    (-> test-folder
        (experiment-util/create-dataset-from-folder class-mapping))))

(defn train-network
  ([argmap] (train-network argmap (train-seq) (test-seq)))
  ([argmap train-ds test-ds]
   (mcc/ensure-images-on-disk!)
   (println "Training for " (:epoch-count argmap) " epochs ")
   (let [listener (classification/create-tensorboard-listener
                   (:tensorboard argmap))]
     (classification/perform-experiment
      (simple-net image-size image-size num-classes)
      train-ds test-ds listener argmap))))

(defn touch-event-file!
  "create the directory and event file to log events. Returns a map with the path "
  [log-path net-name]
  (let [path (str log-path net-name "/tfevents." net-name ".out")]
    (io/make-parents path)
    {:file-path path}))

(defn metric-fn
  [metrics]
  (fn[actual predicted]
    (let [;;run all the metrics 
          scores (mapv (fn [{:keys [mfn fn-name]}]
                         (vector fn-name
                                 (mfn actual predicted))) metrics)
          ;;stream it to tensorboard event file
          _ (doseq [[iname v] scores]
              (eio/make-event iname v))]
      (println "metrics recorded" (clojure.string/join " , " scores )))))

(defn trainres
  [args]
  (let [initargs {:batch-size 128 :epoch-count 2 }
        log-path "/tmp/tflogs/"
        net-name "simple20"
        tfargs {:tensorboard (assoc (touch-event-file! log-path net-name)
                                    :class-mapping class-mapping
                                    :metric-fn
                                    (metric-fn [{:mfn micro-avg-precision 
                                                 :fn-name "precision"}
                                                {:mfn mcm/macro-avg-precision
                                                 :fn-name "macro precision"}
                                                {:mfn accuracy
                                                 :fn-name "accuracy"}]))}
        argmap (merge initargs tfargs)]
    (train-network argmap)))
(comment
  (def tres (trainres {}))
  (-> trainres
      :cv-loss
                                        ; second
      ))

(defn infinite-class-unbalanced-seq
  [map-seq & {:keys [class-key]}]
  (let [ds-map (group-by class-key map-seq)
        key-nine (conj (vec (repeat 9 0.0)) 1.0)
        unbalanced-map  (update-in ds-map [key-nine] #(let [cnt (count %)]
                                         (take (int (* 0.1 cnt)) %)))]
    (println "data " (count unbalanced-map)
             " -- "
             (mapv (fn[[k v]] (str k " count: " (count v))) unbalanced-map))
    (->> unbalanced-map
         (map (fn [[_ v]]
                (->> (repeatedly #(shuffle v))
                     (mapcat identity))))
         (apply interleave))))

(defn infinite-class-unbalanced-dataset
  "Given a dataset, returns an infinite sequence of maps perfectly
  balanced by class."
  [map-seq & {:keys [class-key epoch-size]
              :or {class-key :labels
                   epoch-size 1024}}]
  (->> (infinite-class-unbalanced-seq map-seq :class-key class-key)
       (partition epoch-size)))

(comment 
  (let [initargs {:batch-size 128 :epoch-count 1}
        log-path "/tmp/tflogs/"
        argmap (merge initargs (touch-event-file! log-path "two-conv-layers-unbalanced"))]
    (train-network argmap (train-seq infinite-class-unbalanced-dataset)
                   (test-seq))))
