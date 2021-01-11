import json
import torch

def savejson(model, likelihood, json_file):
    results = {}
    num_task = len(model.mean_module)

    B = model.task_covar_module.covar_factor
    v = model.task_covar_module.var
    K = torch.matmul(B,B.T)+torch.diag(v)
    ls = model.t_covar_module.kernels[0].lengthscale
    noise = likelihood.noise
    weights = []
    biases = []

    for i in range(num_task):
        weight = model.mean_module[i].weights.data.numpy().tolist()
        bias = model.mean_module[i].bias.data.numpy().tolist()
        weights.append(weight)
        biases.append(bias)

    # for i in range(num_task):
    #     weight = model.mean_module.base_means[i].constant.data.numpy().tolist()
    #     weights.append(weight)

    # results['B'] = B.data.numpy().tolist()
    # results['v'] = v.data.numpy().tolist()
    results['task covariance'] = K.data.numpy().tolist()
    results['mean weights'] = weights
    results['mean biases'] = biases
    results['time periodic ls'] = ls.data.numpy().tolist()
    results['noise std'] = torch.sqrt(noise).data.numpy().tolist()

    with open(json_file, "w") as f:
        f.write("{\n")
        for k,v in results.items():
            f.write("\t")
            json.dump(k, f)
            f.write(":")
            json.dump(v, f)
            f.write(",\n")
        f.write('\t"nothing":[]\n')
        f.write("}\n")
