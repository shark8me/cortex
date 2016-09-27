(ns dataframe.core-test
  (:require [clojure.test :refer :all]
            [dataframe.core :as df]))

(defonce test-csv-string
  "float_col,int_col,str_col\n0.1,1,a\n0.2,2,b\n0.2,6,None\n0.1,8,c\nNaN,-1,a")

(deftest basic-api
  (is (= 3 (df/col-count (df/csv->df test-csv-string)))))
