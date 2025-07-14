import torch
from torch.utils.data import Dataset
from typing import List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MedicalQAPair:
    question: str
    answer: str

class MedicalQADataset(Dataset):
    """Medical Question-Answering Dataset"""
    
    def __init__(self, tokenizer, max_length: int = 512, client_id: int = None, total_clients: int = 3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.client_id = client_id
        self.total_clients = total_clients
        self.qa_pairs = self._load_medical_data()
        
    def _load_medical_data(self) -> List[MedicalQAPair]:
        """Load medical QA data from multiple sources"""
        qa_pairs = []
        
        try:
            logger.info("Loading medical QA datasets...")
            qa_pairs = self._load_from_huggingface()
        except Exception as e:
            logger.warning(f"Could not load real dataset: {e}")
            logger.info("Creating synthetic medical QA data...")
            qa_pairs = self._create_sample_medical_data()
        
        # Distribute data among clients
        if self.client_id is not None:
            qa_pairs = self._distribute_data(qa_pairs)
        
        return qa_pairs
    
    def _load_from_huggingface(self) -> List[MedicalQAPair]:
        """Load from HuggingFace medical datasets"""
        qa_pairs = []
        
        # Sample medical QA data templates
        medical_templates = [
            ("What are the symptoms of {disease}?", "The symptoms of {disease} include {symptoms}."),
            ("How is {disease} diagnosed?", "{disease} is diagnosed through {diagnostic_method}."),
            ("What is the treatment for {disease}?", "Treatment for {disease} typically involves {treatment}."),
            ("What causes {disease}?", "{disease} is caused by {cause}."),
            ("How can {disease} be prevented?", "{disease} can be prevented by {prevention}."),
            ("What are the risk factors for {disease}?", "Risk factors for {disease} include {risk_factors}."),
            ("What is the prognosis for {disease}?", "The prognosis for {disease} is generally {prognosis}."),
            ("What are the complications of {disease}?", "Complications of {disease} may include {complications}.")
        ]
        
        diseases = [
            "diabetes", "hypertension", "heart disease", "pneumonia", "migraine", 
            "asthma", "arthritis", "depression", "anxiety", "obesity",
            "stroke", "cancer", "tuberculosis", "hepatitis", "malaria"
        ]
        
        disease_info = {
            "diabetes": {
                "symptoms": "frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision",
                "diagnostic_method": "blood glucose tests, HbA1c tests, and glucose tolerance tests",
                "treatment": "insulin therapy, oral medications, diet control, and regular exercise",
                "cause": "insufficient insulin production or insulin resistance",
                "prevention": "maintaining healthy weight, regular exercise, and balanced diet",
                "risk_factors": "family history, obesity, sedentary lifestyle, and age over 45",
                "prognosis": "good with proper management and lifestyle modifications",
                "complications": "diabetic retinopathy, neuropathy, nephropathy, and cardiovascular disease"
            },
            "hypertension": {
                "symptoms": "headaches, shortness of breath, nosebleeds, chest pain",
                "diagnostic_method": "regular blood pressure measurements and monitoring",
                "treatment": "lifestyle changes, antihypertensive medications, and regular monitoring",
                "cause": "various factors including genetics, lifestyle, and underlying conditions",
                "prevention": "healthy diet, regular exercise, limiting sodium intake, and stress management",
                "risk_factors": "age, family history, obesity, smoking, and excessive alcohol consumption",
                "prognosis": "excellent with proper treatment and lifestyle modifications",
                "complications": "heart attack, stroke, kidney disease, and heart failure"
            }
        }
        
        # Generate QA pairs
        for disease in diseases:
            info = disease_info.get(disease, {
                "symptoms": "various symptoms depending on severity",
                "diagnostic_method": "clinical examination and appropriate tests",
                "treatment": "appropriate medical treatment as prescribed",
                "cause": "multiple factors may contribute",
                "prevention": "healthy lifestyle and regular checkups",
                "risk_factors": "various risk factors may apply",
                "prognosis": "varies depending on individual cases",
                "complications": "potential complications may occur"
            })
            
            for question_template, answer_template in medical_templates:
                for key, value in info.items():
                    if f"{{{key}}}" in answer_template:
                        question = question_template.format(disease=disease)
                        answer = answer_template.format(disease=disease, **{key: value})
                        qa_pairs.append(MedicalQAPair(question=question, answer=answer))
        
        # Expand dataset by repeating with variations
        expanded_pairs = []
        for _ in range(5):  # Repeat 5 times for more data
            expanded_pairs.extend(qa_pairs)
        
        return expanded_pairs
    
    def _create_sample_medical_data(self) -> List[MedicalQAPair]:
        """Fallback: Create sample medical QA data"""
        sample_data = [
            MedicalQAPair(
                question="What are the symptoms of diabetes?",
                answer="Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow-healing wounds."
            ),
            MedicalQAPair(
                question="How is hypertension diagnosed?",
                answer="Hypertension is diagnosed through blood pressure measurements taken on multiple occasions. A reading of 140/90 mmHg or higher is considered high blood pressure."
            ),
            MedicalQAPair(
                question="What are the risk factors for heart disease?",
                answer="Risk factors for heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, family history, and sedentary lifestyle."
            )
        ]
        
        # Expand sample data
        expanded_data = []
        for _ in range(100):
            expanded_data.extend(sample_data)
        
        return expanded_data
    
    def _distribute_data(self, qa_pairs: List[MedicalQAPair]) -> List[MedicalQAPair]:
        """Distribute data among clients (non-IID distribution)"""
        total_samples = len(qa_pairs)
        samples_per_client = total_samples // self.total_clients
        
        start_idx = self.client_id * samples_per_client
        end_idx = start_idx + samples_per_client
        
        # Add some overlap for the last client
        if self.client_id == self.total_clients - 1:
            end_idx = total_samples
        
        return qa_pairs[start_idx:end_idx]
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        
        # Format as instruction-following format
        text = f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels for causal LM (shift by one position)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token for loss calculation
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
