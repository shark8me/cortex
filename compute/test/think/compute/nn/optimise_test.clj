(ns think.compute.nn.optimise-test
  (:require [clojure.test :refer :all]
            [think.compute.verify.optimise :as verify-optimise]
            [think.compute.verify.utils :refer :all]
            [think.compute.nn.cpu-backend :as cpu-net]))

(use-fixtures :each test-wrapper)

(defn create-backend
  []
  (cpu-net/create-cpu-backend *datatype*))

(def-double-float-test adam
  (verify-optimise/test-adam (create-backend)))

(deftest momentum
  (verify-optimise/test-momentum (create-backend)))

(deftest nesterov
  (verify-optimise/test-nesterov (create-backend)))
