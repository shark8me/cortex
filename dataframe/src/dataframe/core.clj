(ns dataframe.core
  (:require [clojure.data.csv :as csv]))

(defn maybe-number
  [s]
  (let [num? (clojure.edn/read-string s)]
    (if (number? num?) num? s)))

(defn csv->df
  [csv & {:keys [:header-row :skip-rows]
          :or {:header-row false
               :skip-rows 0}}]
  (let [rows (csv/read-csv csv)
        rows (drop skip-rows rows)
        [column-labels rows] (if header-row
                               [(first rows) (rest rows)]
                               [[] rows])
        col-count (count (first rows))
        cols (vec (repeatedly col-count #(transient [])))]
    (doseq [row rows]
      (dotimes [i col-count] ;; O(m*n)
        (conj! (get cols i) (maybe-number (get row i)))))
    {:column-labels column-labels
     :col-count col-count
     :cols (mapv persistent! cols)}))

(defn col-count
  [df]
  (:col-count df))

(defn row-count
  [df]
  (-> df :cols first count))

(defn get-col
  [df col]
  (if-let [col-index (cond (string? col) (ffirst (filter #(= col (second %))
                                                         (map-indexed vector (:column-labels df))))
                           (number? col) col)]
    (get (:cols df) col-index)
    (throw (Exception. (format "Could not find index %s in df." col)))))

(defn stats
  ([df]
   (let [col-names (if-not (empty? (:column-labels df))
                     (:column-labels df)
                     (range (:col-count df)))]
     (->> col-names
          (map (fn [name]
                 [name (stats df name)]))
          (into {}))))
  ([df col]
   (let [col (get-col df col)
         types (->> col (map #(if (number? %) :number :string)) frequencies)
         col (filter number? col)]
     (merge
      (loop [n 0
             v (first col)
             s (rest col)
             sum 0.0
             minimum Double/MAX_VALUE
             maximum Double/MIN_VALUE
             types []]
        (if v
          (recur (inc n)
                 (first s)
                 (rest s)
                 (+ sum v)
                 (min minimum v)
                 (max maximum v)
                 (conj types (type v)))
          {:sum sum
           :mean (double (/ sum n))
           :min minimum
           :max maximum}))
      {:types types}))))
