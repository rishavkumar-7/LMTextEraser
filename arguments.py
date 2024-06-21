import argparse
MODEL_LIST = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              "facebook/opt-1.3b",
              "NousResearch/Llama-2-7b-chat-hf"
             ]

def Argument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--use_prompt_tuning",
        action="store_true"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="evaluation_metric",
    )
    parser.add_argument(
        "--_generate_bleu",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--loss_filename",
        type=str,
        default="loss",
    )
    parser.add_argument(
        "--retain_dataset",
        type=str,
        default="truthfulQA",
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=500,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.8, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--retain_weight",
        type=float,
        default=1,
        help="Weight on learning the retain outputs.",
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the retain outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size of unlearning.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Unlearning LR.",
    )

    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/tinyllama-colorist-v1/checkpoint-300/",
        help="Path of the finetuned model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the finetuned model.",
        choices=MODEL_LIST
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        help="base model name to use for unleaning ",
        choices=MODEL_LIST
    )


    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/tinyllama_unlearned_color/",
        help="Directory to save model.",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="How many steps to save model.",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(args)