from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

training_data = "training_data/CGN/cgn_sample_10000.txt"  # Replace [...] with your actual training data
base_name = os.path.basename(training_data).split('.')[0]
output_dir = f'finetuned_models/./finetuned_model_{base_name}'

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')

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