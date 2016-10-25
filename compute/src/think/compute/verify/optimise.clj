(ns think.compute.verify.optimise
  (:require [clojure.test :refer :all]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.datatype :as dtype]
            [think.compute.optimise :as opt]
            [think.compute.verify.utils :refer [def-double-float-test] :as utils]
            [clojure.core.matrix :as m]))


(def adam-answers
  [[0.999000000005, 1.9990000000025, 2.9990000000016668, 3.99900000000125]
   [0.9980000262138343, 1.9980000130723587, 2.998000008707231, 3.998000006527546]
   [0.9970000960651408, 1.9970000478972731, 2.9970000319014805, 3.9970000239148513]
   [0.9960002269257634, 1.9960001131228113, 2.9960000753397424, 3.9960000564765217]
   [0.995000436052392, 1.9950002173334274, 2.995000144735273, 3.995000108493863]
   [0.9940007405541528, 1.9940003690339771, 2.994000245746984, 3.9940001842069432]
   [0.9930011573564278, 1.9930005766318695, 2.9930003839675967, 3.993000287805732]
   [0.9920017031661642, 1.9920008484199827, 2.992000564912314, 3.992000423421624]
   [0.9910023944389119, 1.991001192560463, 2.9910007940080785, 3.9910005951194076]
   [0.9900032473478027, 1.9900016170695067, 2.9900010765834972, 3.990000806889731]
   [0.9890042777546556, 1.9890021298032192, 2.9890014178594777, 3.9890010626421075]
   [0.9880055011833645, 1.9880027384446224, 2.9880018229406398, 3.988001366198499]
   [0.9870069327956914, 1.987003450491873, 2.98700229680753, 3.9870017212875037]
   [0.9860085873695607, 1.986004273247733, 2.9860028443096756, 3.9860021315391725]
   [0.9850104792799151, 1.9850052138103236, 2.9850034701594894, 3.9850026004804673]
   [0.9840126224821667, 1.984006279065173, 2.984004178927043, 3.984003131531364]
   [0.9830150304982433, 1.9830074756785594, 2.9830049750356973, 3.9830037280016044]
   [0.9820177164052082, 1.9820088100921336, 2.982005862758586, 3.9820043930880837]
   [0.9810206928263993, 1.9810102885187928, 2.981006846215933, 3.9810051298728633]])


(defn test-adam
  [backend]
  (let [parameters (nn-backend/array backend [1 2 3 4])
        optimizer (opt/setup-optimiser (opt/adam) backend 4)]
    (reduce (fn [optimizer item]
              (let [gradient (nn-backend/array backend (map #(* 2.0 %) (nn-backend/to-double-array backend parameters)))
                    optimizer (opt/batch-update optimizer)
                    _ (opt/compute-parameters! optimizer 1.0 0 gradient parameters)
                    item (m/eseq item)
                    guess (m/eseq (nn-backend/to-double-array backend parameters))]
                (is (utils/about-there? guess item))
                optimizer))
            optimizer
            adam-answers)))

(defn test-momentum
  [backend]
  (let [df #(* 2.0 %)
        parameters (nn-backend/array backend [8.0])
        optimizer (opt/setup-optimiser (opt/momentum) backend 1)
        final-param-vec (loop [i 1000]
                          (if (zero? i)
                            (->> parameters (nn-backend/to-double-array backend) vec)
                            (let [gradient (nn-backend/array backend (map df (nn-backend/to-double-array backend parameters)))]
                              (opt/compute-parameters! optimizer 1.0 0 gradient parameters)
                              (recur (dec i)))))]
    #_(println final-param-vec)
    (is (every? #(< % 1e-6) final-param-vec))))

(defn test-nesterov
  "Like momentum, but we modify the params before calculating the gradient.
  See: http://sebastianruder.com/optimizing-gradient-descent/index.html#nesterovacceleratedgradient"
  [backend]
  (let [df #(* 2.0 %)
        parameters (nn-backend/array backend [8.0])
        optimizer (opt/setup-optimiser (opt/momentum) backend 1)
        final-param-vec (loop [i 1000]
                          (if (zero? i)
                            (->> parameters (nn-backend/to-double-array backend) vec)
                            (let [gamma (:gamma (.options optimizer))
                                  v-sub-t-1-vec (->> optimizer :v-sub-t-1 (nn-backend/to-double-array backend) vec)
                                  param-vec (->> parameters (nn-backend/to-double-array backend) vec)
                                  ;; TODO: This depends on the vecs being the same length (incompatible w/ multiple layers)
                                  ;;       Perhaps fixable by creating an appropriate view (subvec) into v-sub-t-1?
                                  nesterov-params (mapv (fn [param last-update]
                                                          (- param (* gamma last-update)))
                                                        param-vec v-sub-t-1-vec)
                                  ;; Note: Here df is a pure function of the parameters, in a neural net this means (I think)
                                  ;;       speculatively updating the weights and running the net forward again (to get an
                                  ;;       output -> error -> gradient).
                                  ;; Thought: This results in a doubling of the number of forward passes, but could still
                                  ;;          result in better training per unit time if the network converges faster.
                                  gradient (nn-backend/array backend (map df nesterov-params))]
                              (opt/compute-parameters! optimizer 1.0 0 gradient parameters)
                              (recur (dec i)))))]
    ;; Interestingly, in this simple case nesterov converges slightly _slower_ than normal momentum.
    ;; When the loss is decreasing monotonically, that is when the params are haeded in the right
    ;; direction, nesterov's looking ahead results in _smaller_ gradients and thus slower convergence.
    ;; Nesterov could be more stable at higher learning rates which could make up for this.
    ;; Seems testing on functions other than 1d (* x x) could prove the value better.
    #_(println final-param-vec)
    (is (every? #(< % 1e-6) final-param-vec))))
