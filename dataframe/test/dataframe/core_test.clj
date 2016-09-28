(ns dataframe.core-test
  (:require [clojure.test :refer :all]
            [dataframe.core :as df]))

(defonce test-csv-string
  "float_col,int_col,str_col\n0.1,1,a\n0.2,2,b\n0.2,6,None\n0.1,8,c\nNaN,-1,a")

(deftest basic-api
  (let [dataframe (df/csv->df test-csv-string :header-row true)]
    (is (= 3 (df/col-count dataframe)))
    (is (= 5 (df/row-count dataframe)))
    (let [cols (df/cols dataframe)
          rows (df/rows dataframe)]
      (is (every? vector? cols))
      (is (every? vector? rows)))
    (let [stats (df/stats dataframe)]
      (is (map? stats))
      (is (= (count stats) (df/col-count dataframe))))))
