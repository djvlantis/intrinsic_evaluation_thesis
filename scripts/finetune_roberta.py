from transformers import pipeline
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

training_data = "training_data/CGN/cgn_sample_10000.txt" 
base_name = os.path.basename(training_data).split('.')[0]
output_dir = f'finetuned_models/./finetuned_roberta_model_{base_name}'

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=training_data, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
)   

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Fine-tuned model saved successfully.")
