(ns cortex.suite.train_test
  (:require [clojure.test :refer :all]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.suite.train :as ctrain]
            [cortex.nn.traverse :as traverse]
            [clojure.core.matrix :as m]))


(ctrain/save-network
  
  )

(ctrain/create-context)
(suite-train/train-n dataset initial-description network
                                    :best-network-fn (partial best-network-function
                                                              confusion-matrix-atom
                                                              observation->image-fn
                                                              dataset)
                                    :epoch-count epoch-count
                                    :batch-size batch-size)

(let [init-desc "trial"
      network (-> (network/build-network init-desc)
                    traverse/auto-bind-io)]
  network
   #_(doseq [_ (repeatedly
              #(suite-train/train-n dataset initial-description network
                                    :best-network-fn (partial best-network-function
                                                              confusion-matrix-atom
                                                              observation->image-fn
                                                              dataset)
                                    :epoch-count epoch-count
                                    :batch-size batch-size))]))
