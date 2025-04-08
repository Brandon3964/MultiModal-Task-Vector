from mtv_utils import *
from models import *
from preprocess import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 


def eval_reinforce(args):

    train_dataset = open_data(args.data_name, args.train_path)
    val_dataset = open_data(args.data_name, args.val_path)


    activation_data = train_dataset
    reinforce_data = train_dataset #random.sample(train_dataset, 100)
    eval_data = val_dataset#[:50]

    ##Load the model
    model_helper = load_model(args.model_name, args.data_name)

    ##Mean activation of some in-context input
    if args.cur_mode != "clean":

        mean_activations = get_last_mean_head_activations(activation_data, model_helper, N_TRIALS = args.num_example, shot=args.num_shot)

        torch.save(mean_activations, args.activation_path)
        mean_activations = torch.load(args.activation_path)

        # ##Examples from the test set is used to visualize the validation loss
        bernoullis = reinforce(mean_activations, model_helper, reinforce_data, eval_data)
        # torch.save(bernoullis, args.bernoullis_path)
        # bernoullis = torch.load(args.bernoullis_path)

        best_heads = (999, None)
        ###Sample multiple times and pick the best set of heads.
        for _ in range(10):
            ###Sample from the trained distribution and identify the intervention locations
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis])
            ###Thresholding heads with low probability from being sampled. Reduce the number of heads. Idefics2 empirically benefit from less heads.
            if args.model_name == "idefics2":
                sigmoid_tensor = torch.nn.functional.threshold(sigmoid_tensor, 0.8, 0)

            
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled = prob_dist.sample()
            intervention_locations = reinforce_intervention_location(sampled)
            cur_heads_loss = validate_reinforce(model_helper, bernoullis, 1e-3, mean_activations, train_dataset[:50], 0, sampled=sampled)
            if cur_heads_loss < best_heads[0]:
                best_heads = (cur_heads_loss, intervention_locations)
        torch.save(best_heads[1], args.bernoullis_path)
        intervention_locations = best_heads[1]

        intervention_locations = torch.load(args.bernoullis_path)
    else:
        mean_activations = None
        intervention_locations = None

    clean_answers = []
    interv_answers = []

    # Create lists to store the test data for later group processing
    clean_test_data = []
    interv_test_data = []

    for item in tqdm(val_dataset):
        text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=args.eval_num_shot)
        new_input = model_helper.insert_image(text, image_list)
        clean_out, interv_out = fv_intervention_natural_text(new_input, model_helper, max_new_tokens=args.max_token, return_item=args.cur_mode, intervention_locations=intervention_locations, avg_activations=mean_activations)
        
        # print(f'clean_out {clean_out}')
        # print(f'interv out {interv_out}')
        # Add to score lists
        interv_answers.append({"output": interv_out, "question_id": question_id}) # Can change this to have an index for query and context
        clean_answers.append({"output": clean_out, "question_id": question_id})
        
        
        # # Create entries for NaturalBench-style group processing
        # clean_test_data.append({
        #     "image": image_list[0],  # Assuming single image per item
        #     "question": text,
        #     "question_id": question_id,
        #     "label": target_out,
        #     "pred": "Yes" if clean_out >= .5 else "No"
        # })
        
        # interv_test_data.append({
        #     "image": image_list[0],  # Assuming single image per item
        #     "question": text,
        #     "question_id": question_id,
        #     "label": target_out,
        #     "pred": "Yes" if interv_out >= .5 else "No"
        # })
    with open('/home/chancharikm/FSOD_data/wood_defects_clean.json', 'w') as writefile:
        for line in clean_answers:
            writefile.write(json.dumps(line) + '\n')
    with open('/home/chancharikm/FSOD_data/wood_defects_interv.json', 'w') as writefile:
        for line in interv_answers:
            writefile.write(json.dumps(line) + '\n')
    
    # if args.is_eval:
    #     # Process and evaluate based on mode
    #     if args.cur_mode == "interv" or args.cur_mode == "both":
    #         if args.data_name == "natural_ret":
    #             print(f"\nNaturalBench Detailed Metrics for Intervention:")
    #             evaluate_naturalbench(interv_test_data)
    #         # elif args.data_name == "wino"

    #     if args.cur_mode == "clean" or args.cur_mode == "both":
    #         # NaturalBench evaluation for clean answers
    #         if args.data_name == "natural_ret":
    #             print(f"\nNaturalBench Detailed Metrics for Clean:")
    #             evaluate_naturalbench(clean_test_data)
    #         # elif args.data_name
    # with open('./storage/' + args.data_name + '_clean_scores.jsonl', 'w') as writefile:
    #     for line in clean_answers:
    #         writefile.write(json.dumps(line) + '\n')
    # with open('./storage/' + args.data_name + '_interv_scores.jsonl', 'w') as writefile:
    #     for line in interv_answers:
    #         writefile.write(json.dumps(line) + '\n')

    def evaluate_naturalbench(test_data):
        """
        Evaluate NaturalBench dataset using the group-based metrics.
        """
        q_correct = 0  # Question accuracy count
        i_correct = 0  # Image accuracy count  
        g_correct = 0  # Group accuracy count
        correct = 0  # Raw accuracy count
        total_groups = len(test_data) // 4  # Total number of groups
        
        # Process data in groups of 4
        for i in range(0, len(test_data), 4):
            group = test_data[i:i+4]
            group_preds = []
            
            # Get predictions for the group
            for item in group:
                pred = item["pred"]
                label = item["label"]
                group_preds.append(pred.lower() == label.lower())
            
            # Question accuracy (first two and second two must match)
            if group_preds[0] and group_preds[1]:
                q_correct += 1
            if group_preds[2] and group_preds[3]:
                q_correct += 1
                
            # Image accuracy (first and third, second and fourth must match)
            if group_preds[0] and group_preds[2]:
                i_correct += 1
            if group_preds[1] and group_preds[3]:
                i_correct += 1
                
            # Group accuracy (all four must be correct)
            if all(group_preds):
                g_correct += 1

            # Raw accuracy
            correct += sum(group_preds)

        # Calculate percentages
        q_acc = q_correct / (total_groups * 2)  # Two questions per group
        i_acc = i_correct / (total_groups * 2)  # Two images per group
        g_acc = g_correct / total_groups        # One group accuracy score per group
        acc = correct / (total_groups * 4)      # Accuracy calculated per sample
        
        print(f"Question accuracy: {q_acc:.4f}")
        print(f"Image accuracy: {i_acc:.4f}")
        print(f"Group accuracy: {g_acc:.4f}")
        print(f"Raw accuracy: {acc:.4f}")
        
        return {
            "question_accuracy": q_acc,
            "image_accuracy": i_acc,
            "group_accuracy": g_acc,
            "raw_accuracy": acc
        }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen")
    parser.add_argument("--data_name", type=str, default="vizwiz")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--num_example", type=int, default=100)
    parser.add_argument("--num_shot", type=int, default=4)
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--bernoullis_path", type=str, default=None)
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--activation_path", type=str, default=None)
    
    args = parser.parse_args()

    eval_reinforce(args)

