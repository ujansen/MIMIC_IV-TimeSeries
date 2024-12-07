import re
import ast
import json


def extract_values(namespace_line):
    train_frac_match = re.search(r'train_frac:(\S+?)(\|)', namespace_line)
    run_match = re.search(r'run:(\S+)', namespace_line)
    train_frac = train_frac_match.group(1) if train_frac_match else None
    run = run_match.group(1) if run_match else None
    return train_frac, run

def extract_metrics(metric_line):
    try:
        metrics_dict = ast.literal_eval(metric_line.split(":", 1)[1].strip())
        return {
            "auroc": metrics_dict.get("auroc"),
            "auprc": metrics_dict.get("auprc"),
            "minrp": metrics_dict.get("minrp"),
            "accuracy": metrics_dict.get("accuracy")
        }
    except (ValueError, SyntaxError):
        return {
            "auroc": None,
            "auprc": None,
            "minrp": None,
            "accuracy": None
        }

def process_log_file(file_path):
    data_collection = []
    namespace, final_val_res, final_test_res, prepare = None, None, None, None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if ">> Namespace" in line and "pretrain=0" in line:
                namespace = line.split(">>", 1)[1].strip()
                train_frac, run = extract_values(namespace)
            elif ">> Preparing dataset mimic_iv for training" in line:
                prepare = line.split(">>", 1)[1].strip()
            elif ">> Final val res" in line and "auroc" in line:
                final_val_res = line.split(">>", 1)[1].strip()
                val_metrics = extract_metrics(final_val_res)
            elif ">> Final test res" in line and "auroc" in line:
                final_test_res = line.split(">>", 1)[1].strip()
                test_metrics = extract_metrics(final_test_res)

                if namespace and prepare and final_val_res:
                    data_collection.append({
                        'Namespace': namespace,
                        'Train Fraction': train_frac,
                        'Run': run,
                        'Final val res': val_metrics,
                        'Final test res': test_metrics
                    })

                    namespace = None
                    final_test_res = None
                    final_val_res = None
                    prepare = None
    
    return data_collection

file_path = './strats_output_14066025.log'
results = process_log_file(file_path)

consolidated_results = {}

for entry in results:
    train_frac = entry['Train Fraction']
    run = entry['Run']
    if train_frac not in consolidated_results:
        consolidated_results[train_frac] = {}
    consolidated_results[train_frac][run] = {
        'val': entry['Final val res'],
        'test': entry['Final test res']
    }

json_file = 'consolidated_data.json'
with open(json_file, 'w') as file:
    json.dump(consolidated_results, file, indent=4)

