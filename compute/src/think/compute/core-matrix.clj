(ns think.compute.core-matrix
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as cp]
            [think.compute.driver :as drv]
            [think.compute.datatype :as dtype]
            [think.compute.math :as c-math]))



(def ^:dynamic *driver* nil)
(def ^:dynamic *stream* nil)
(def ^:dymamic *datatype* :double)
