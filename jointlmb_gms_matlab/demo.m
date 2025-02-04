% This is the demo script for the Labeled Multi-Bernoulli filter proposed in
% S. Reuter, B.-T. Vo, B.-N. Vo, and K. Dietmayer, "The labelled multi-Bernoulli filter," IEEE Trans. Signal Processing, Vol. 62, No. 12, pp. 3246-3260, 2014
% http://ba-ngu.vo-au.com/vo/RVVD_LMB_TSP14.pdf
% which propagates an LMB approximation of the GLMB update proposed in
% B.-T. Vo, and B.-N. Vo, "Labeled Random Finite Sets and Multi-Object Conjugate Priors," IEEE Trans. Signal Processing, Vol. 61, No. 13, pp. 3460-3475, 2013.
% http://ba-ngu.vo-au.com/vo/VV_Conjugate_TSP13.pdf
% and
% B.-N. Vo, B.-T. Vo, and D. Phung, "Labeled Random Finite Sets and the Bayes Multi-Target Tracking Filter," IEEE Trans. Signal Processing, Vol. 62, No. 24, pp. 6554-6567, 2014
% http://ba-ngu.vo-au.com/vo/VVP_GLMB_TSP14.pdf
% using an efficient implementation of the GLMB filter proposed in
% B.-T. Vo, and B.-N. Vo, "An Efficient Implementation of the Generalized Labeled Multi-Bernoulli Filter," IEEE Trans. Signal Processing, Vol. 65, No. 8, pp. 1975-1987, 2017.
% http://ba-ngu.vo-au.com/vo/VVH_FastGLMB_TSP17.pdf
%
%
% Note 1: no dynamic grouping or adaptive birth is implemented in this code, only the standard filter with static birth is given
% Note 2: the simple example used here is the same as in the CB-MeMBer filter code for a quick demonstration and comparison purposes
% Note 3: more difficult scenarios require more components/hypotheses (thus exec time)
% ---BibTeX entry
% @ARTICLE{LMB,
% author={S. Reuter and B.-T. Vo and B.-N. Vo and K. Dietmayer},
% journal={IEEE Transactions on Signal Processing},
% title={The Labeled Multi-Bernoulli Filter},
% year={2014},
% month={Jun}
% volume={62},
% number={12},
% pages={3246-3260}}
%
% @ARTICLE{GLMB1,
% author={B.-T. Vo and B.-N. Vo
% journal={IEEE Transactions on Signal Processing},
% title={Labeled Random Finite Sets and Multi-Object Conjugate Priors},
% year={2013},
% month={Jul}
% volume={61},
% number={13},
% pages={3460-3475}}
%
% @ARTICLE{GLMB2,
% author={B.-T. Vo and B.-N. Vo and D. Phung},
% journal={IEEE Transactions on Signal Processing},
% title={Labeled Random Finite Sets and the Bayes Multi-Target Tracking Filter},
% year={2014},
% month={Dec}
% volume={62},
% number={24},
% pages={6554-6567}}
%
% @ARTICLE{GLMB3,
% author={B.-N. Vo and B.-T. Vo and H. Hung},
% journal={IEEE Transactions on Signal Processing},
% title={An Efficient Implementation of the Generalized Labeled Multi-Bernoulli Filter},
% year={2017},
% month={Apr}
% volume={65},
% number={8},
% pages={1975-1987}}
%---

model= gen_model;
truth= gen_truth(model);
meas=  gen_meas(model,truth);
est=   run_filter(model,meas);
handles= plot_results(model,truth,meas,est);