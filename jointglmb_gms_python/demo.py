from gen_model import model
from gen_truth import truth
from gen_meas import meas
from plot_results import plot_results
from run_filter import GLMB, filter, est

if __name__ == '__main__':
    model_params = model();
    truth_params = truth(model_params);
    meas_params = meas(model_params, truth_params);
    glmb = GLMB();
    filter_params = filter(model_params)
    est_params = est(meas_params)

    glmb.run(model_params, filter_params, meas_params, est_params)
    # glmb.plot_tracks(model_params, filter_params, truth_params, meas_params)

    handles= plot_results(model_params,truth_params,meas_params,est_params)
