(ns suite-classification.main
  (:gen-class))

(defn -main
  [& args]
  (let [argcount (count args)
        train-count (if (>= argcount 1)
                      (try (Integer/parseInt (first args))
                           (catch Throwable e
                             (println
                              (format "train iteration count unparseable %s" (first args))
                              nil)))
                      nil)]

    (require 'suite-classification.core)
    ((resolve 'suite-classification.core/train) train-count)))
