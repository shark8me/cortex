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
  ([argmap] (train-network (train-seq) (test-seq) argmap ))
  ([train-ds test-ds argmap ]
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
          _ (println (str "act " (vec (take 2 actual)) " pred " (vec (take 2 predicted))))
          evs (mapv (fn[[iname v]]
                      (eio/make-event iname v) )scores)]
      (println "metrics recorded" (clojure.string/join " , " scores ))
      evs)))

(defn per-class-metric-fn
  [metric]
  (fn[actual predicted]
    (let [;;run all the metrics
          pairs (mapv (fn[i] (filterv (fn[[a b]] (= i a))
                                      (mapv vector actual predicted)))
                      (mapv str (range 9)))
          kfn (fn[i] [(mapv first i) (mapv second i)])
          scores (mapv #(apply metric (kfn %)) pairs)
          
          ;;stream it to tensorboard event file
          evs (mapv #(eio/make-event %1 %2)
                    (mapv str (range 9))
                    scores)
          ]
      (println "metrics recorded" (clojure.string/join " , " scores ))
      evs)))

(def metr-3
  (metric-fn [{:mfn micro-avg-precision 
               :fn-name "precision"}
              {:mfn mcm/macro-avg-precision
               :fn-name "macro precision"}
              {:mfn accuracy
               :fn-name "accuracy"}]))

(defn trainres
  ([] (trainres train-network))
  ([train-fn]
   (let [log-path "/tmp/tflogs/"
         net-name "simple201"
         tfargs 
         {:tensorboard (assoc
                        (touch-event-file! log-path net-name)
                        :class-mapping class-mapping
                        ;:metric-fn metr-3
                        :metric-fn (per-class-metric-fn accuracy)
                        )}]
     (trainres tfargs train-fn)))
  ([tfargs train-fn]
   (let [initargs {:batch-size 128 :epoch-count 2 }
         argmap (merge initargs tfargs)]
     (train-fn argmap))))

(comment
  (def tres
    (try (trainres)
         (catch Exception e
           (println " caught excception " e))))
  )

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

(defn trainres2
  []
  (let [log-path "/tmp/tflogs/"
        net-name "simple20"
        tfargs 
        {:tensorboard (assoc
                       (touch-event-file! log-path net-name)
                       :class-mapping class-mapping
                       :metric-fn metr-3)}]
    (trainres tfargs (partial
                      train-network
                      (train-seq infinite-class-unbalanced-dataset)
                              (test-seq)))))
(comment
  (def tres
    (try (trainres2)
         (catch Exception e
           (println " caught excception " e))))
  )
