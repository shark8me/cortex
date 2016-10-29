(ns think.compute.core-matrix
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as cp]
            [think.compute.driver :as drv]
            [think.compute.datatype :as dtype]
            [think.compute.math :as c-math])
  (:import [think.compute.math DeviceArray Tensor]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(def ^:dynamic *compute-binding* {:driver nil
                                  :stream nil
                                  :datatype nil})

(defn driver [] (get :driver *compute-binding*))
(defn stream [] (get :stream *compute-binding*))
(defn datatype [] (get :datatype *compute-binding*))

(def default-impl (c-math/->DeviceArray nil nil))

(defn tensor-indexes->aget-idx
  ^long [^Tensor t indexes]
  (let [stride-h (.width t)
        stride-c (* stride-h (.height t))
        stride-b (* stride-c (.channel-count t))
        num-indexes (count indexes)
        strides [1 stride-h stride-c stride-b]]
    (loop [idx 0
           accum 0]
      (if (< idx num-indexes)
        (let [rel-idx (- idx num-indexes 1)]
          (recur
           (inc idx)
           (+ accum
              (* (long (get strides idx))
                 (long (get indexes rel-idx))))))
        accum))))

(defn ->host
  [ary]
  (let [host-buffer (drv/allocate-host-buffer (driver) (m/ecount ary)
                                              (dtype/get-datatype ary))]
    (drv/copy-device->host (stream) (c-math/device-buffer ary) 0 host-buffer 0
                           (m/ecount ary))
    (drv/wait-for-event (drv/create-event (stream)))
    host-buffer))

(defn host->device-array
  [host-buf ary]
  (drv/copy-host->device (stream) host-buf 0
                         (c-math/device-buffer ary) 0
                         (m/ecount ary)))


(extend-type DeviceArray
  cp/PImplementation
  (implementation-key [m] :think-compute)
  (meta-info [m] {:doc "core matrix implementation for the think-compute abstraction"})
  (construct-matrix [m data] (c-math/array (driver) (stream) (datatype) data))
  (new-vector [m len] (c-math/new-array (driver) (stream) (datatype) [len]))
  (new-matrix [m rows columns] (c-math/new-array (driver) (stream) (datatype) [rows columns]))
  (new-matrix-nd [m shape] (c-math/new-array (driver) (stream) (datatype) shape))
  (supports-dimensionality? [m dimensions] (< (long dimensions) 4))

  cp/PDimensionInfo
  (dimensionality [m]
    (let [^Tensor tensor (.tensor m)]
      (if (= (.batch-size tensor) 1)
        (if (= (.channel-count tensor) 1)
          (if (= (.height tensor) 1)
            1
            2)
          3)
        4)))
  (get-shape [m]
    (let [^Tensor tensor (.tensor m)
          dims (long (cp/dimensionality m))]
      (case dims
        1 [(.width tensor)]
        2 [(.height tensor) (.width tensor)]
        3 [(.channel-count tensor) (.height tensor) (.width tensor)]
        4 [(.batch-size tensor) (.channel-count tensor) (.height tensor) (.width tensor)])))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [^Tensor t (.tensor m)]
     (case (long dimension-number)
       1 (.width t)
       2 (.height t)
       3 (.channel-count t)
       4 (.batch-size t)
       (throw (Exception. "Untenable number of dimensions!")))))

  cp/PIndexedAccess
  (get-1d [m row]
    (when-not (= 1 (cp/dimensionality m))
      (throw (Exception. (format
                          "Cannot do 1d access of dimensionalty %s array"
                          (cp/dimensionality m)))))
    (dtype/get-value (->host m) row))
  (get-2d [m row column]
    (when-not (= 2 (cp/dimensionality m))
      (throw (Exception. (format
                          "Cannot do 2d access of dimensionality %s array"
                          (cp/dimensionality m)))))
    (let [stride (.width ^Tensor (.tensor m))]
      (dtype/get-value (->host m) (+ (* (long row) stride) (long column)))))

  (get-nd [m indexes]
    (when-not (= (count indexes) (cp/dimensionality m))
      (throw (Exception. (format
                          "Dimensionality mismatch, index-count %s dimensionality %s"
                          (count indexes) (cp/dimensionality m)))))
    (let [aget-idx (tensor-indexes->aget-idx (.tensor m) indexes)]
      (dtype/get-value (->host m) aget-idx)))

  cp/PMatrixCloning
  (clone [m]
    (let [device-buf (drv/allocate-device-buffer (driver) (m/ecount m) (datatype))]
      (drv/copy-device->device (stream) (c-math/device-buffer m) 0
                               device-buf 0 (m/ecount m))
      (c-math/->DeviceArray device-buf (.tensor m))))

  cp/PIndexedSettingMutable
  (set-1d! [m row v]
    (let [host-buf (->host m)]
      (dtype/set-value! host-buf row v)
      (host->device-array host-buf m)))
  (set-2d! [m row column v]
    (let [stride (.width ^Tensor (.tensor m))
          host-buf (->host m)]
      (dtype/set-value! host-buf (+ (* (long row) stride) (long column)) v)
      (host->device-array host-buf m)))
  (set-nd! [m indexes v]
    (let [aget-idx (tensor-indexes->aget-idx (.tensor m) indexes)
          host-buf (->host m)]
      (dtype/set-value! host-buf aget-idx v)
      (host->device-array host-buf m)))

  cp/PIndexedSetting
  (set-1d [m row v] (cp/set-1d! (cp/clone m) row v))
  (set-2d [m row column v] (cp/set-2d! (cp/clone m) row column v))
  (set-nd [m indexes v] (cp/set-nd! (cp/clone m) indexes v))
  (is-mutable? [m] true)

  cp/PRowColMatrix
  (column-matrix [m data]
    (c-math/as-column-vector (cp/construct-matrix m data)))
  (row-matrix [m data]
    (c-math/as-row-vector (cp/construct-matrix m data)))

  cp/PMutableMatrixConstruction
  (mutable-matrix [m] (cp/clone m))

  cp/PMutableCoercion
  (ensure-mutable [m] m)

  cp/PNative
  (native [m] (c-math/to-double-array (driver) (stream) ary))
  (native? [m] false)

  cp/PDense
  (dense-coerce [m data] (cp/construct-matrix m data))
  (dense [m] m)


  cp/PAssignment
  (assign! [m source]
    (c-math/assign! (stream) m source)
    m)
  (assign-array!
    ([m arr]
     (cp/assign-array! m arr 0 (m/ecount arr)))
    ([m arr start length]
     (let [host-buf (->host m)]
       (dtype/copy! arr start host-buf start length)
       (host->device-array host-buf m)
       m)))

  cp/PImmutableAssignment
  (assign [m source]
    (cp/assign! (cp/clone m) source))

  cp/PDoubleArrayOutput
  (to-double-array [m] (c-math/to-double-array (driver) (stream) m))
  (as-double-array [m] nil))
