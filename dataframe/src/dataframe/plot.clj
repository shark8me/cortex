(ns dataframe.plot
  (:require [mikera.image.core :as i])
  (:import [org.jfree.chart ChartFactory]
           [org.jfree.data.category DefaultCategoryDataset]))

(defn plot
  [sequence-of-numbers & {:keys [:w :h :title :x-label :y-label :data-label]
                          :or {:w 800
                               :h 400
                               :title "Plot"
                               :x-label "x"
                               :y-label "y"
                               :data-label "Unknown"}}]
  (let [ds (DefaultCategoryDataset.)]
    (doseq [[x y] (map-indexed vector sequence-of-numbers)]
      (.addValue ds y data-label x))
    (-> (ChartFactory/createLineChart title x-label y-label ds)
        (.createBufferedImage w h))))

(comment

  (-> (plot (repeatedly 50 #(+ 1 (rand))))
      (i/save "plot.png"))

  )
