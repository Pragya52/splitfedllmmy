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
    """Medical Question-Answering Dataset with Real HuggingFace Integration"""
    
    def __init__(self, tokenizer, max_length: int = 512, client_id: int = None, total_clients: int = 3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.client_id = client_id
        self.total_clients = total_clients
        self.qa_pairs = self._load_medical_data()
        
    def _load_medical_data(self) -> List[MedicalQAPair]:
        """Load medical QA data with proper fallback mechanism"""
        qa_pairs = []
        
        # Try loading real datasets first
        try:
            logger.info("Attempting to load real medical QA datasets from HuggingFace...")
            qa_pairs = self._load_real_huggingface_data()
            logger.info(f"Successfully loaded {len(qa_pairs)} real medical QA pairs")
        except Exception as e:
            logger.warning(f"Could not load real dataset: {e}")
            logger.info("Falling back to synthetic medical QA data...")
            qa_pairs = self._create_synthetic_medical_data()
            logger.info(f"Generated {len(qa_pairs)} synthetic medical QA pairs")
        
        # Distribute data among clients
        if self.client_id is not None:
            qa_pairs = self._distribute_data(qa_pairs)
            logger.info(f"Client {self.client_id} received {len(qa_pairs)} QA pairs")
        
        return qa_pairs
    
    def _load_real_huggingface_data(self) -> List[MedicalQAPair]:
        """Load real medical datasets from HuggingFace"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library not installed. Install with: pip install datasets")
        
        qa_pairs = []
        
        # Try multiple medical QA datasets
        dataset_configs = [
            {
                "name": "medmcqa",
                "config": None,
                "question_col": "question",
                "answer_col": "exp",  # explanation as answer
                "max_samples": 1000
            },
            {
                "name": "pubmed_qa",
                "config": "pqa_labeled",
                "question_col": "question", 
                "answer_col": "long_answer",
                "max_samples": 500
            }
        ]
        
        for dataset_config in dataset_configs:
            try:
                logger.info(f"Loading {dataset_config['name']} dataset...")
                
                # Load dataset
                if dataset_config["config"]:
                    dataset = load_dataset(dataset_config["name"], dataset_config["config"])
                else:
                    dataset = load_dataset(dataset_config["name"])
                
                # Get train split
                train_data = dataset["train"] if "train" in dataset else dataset["validation"]
                
                # Convert to our format
                max_samples = min(len(train_data), dataset_config["max_samples"])
                for i in range(max_samples):
                    item = train_data[i]
                    
                    # Extract question and answer
                    question = item.get(dataset_config["question_col"], "")
                    answer = item.get(dataset_config["answer_col"], "")
                    
                    # Skip if empty
                    if question and answer:
                        qa_pairs.append(MedicalQAPair(
                            question=str(question).strip(),
                            answer=str(answer).strip()
                        ))
                
                logger.info(f"Loaded {len(qa_pairs)} pairs from {dataset_config['name']}")
                
            except Exception as e:
                logger.warning(f"Failed to load {dataset_config['name']}: {e}")
                continue
        
        # If we got some real data, return it
        if qa_pairs:
            return qa_pairs
        else:
            # If no real data loaded, raise exception to trigger fallback
            raise Exception("No real medical datasets could be loaded")
    
    def _create_synthetic_medical_data(self) -> List[MedicalQAPair]:
        """Create synthetic medical QA data when real data unavailable"""
        qa_pairs = []
        
        # Medical QA templates
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
            "stroke", "cancer", "tuberculosis", "hepatitis", "malaria",
            "alzheimer's disease", "parkinson's disease", "epilepsy", "osteoporosis", "copd"
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
            },
            "heart disease": {
                "symptoms": "chest pain, shortness of breath, fatigue, irregular heartbeat",
                "diagnostic_method": "ECG, echocardiogram, stress tests, and cardiac catheterization",
                "treatment": "medications, lifestyle changes, surgical procedures, and cardiac rehabilitation",
                "cause": "atherosclerosis, high blood pressure, smoking, and genetic factors",
                "prevention": "healthy diet, regular exercise, smoking cessation, and stress management",
                "risk_factors": "high cholesterol, diabetes, smoking, family history, and age",
                "prognosis": "varies depending on type and severity of condition",
                "complications": "heart attack, heart failure, arrhythmias, and sudden cardiac death"
            }
        }
        
        # Generate QA pairs
        for disease in diseases:
            info = disease_info.get(disease, {
                "symptoms": "various symptoms depending on severity and individual factors",
                "diagnostic_method": "clinical examination, medical history, and appropriate diagnostic tests",
                "treatment": "appropriate medical treatment as prescribed by healthcare professionals",
                "cause": "multiple factors may contribute including genetic and environmental influences",
                "prevention": "healthy lifestyle choices and regular medical checkups",
                "risk_factors": "various risk factors may apply depending on individual circumstances",
                "prognosis": "varies depending on individual cases and treatment response",
                "complications": "potential complications may occur if left untreated"
            })
            
            for question_template, answer_template in medical_templates:
                for key, value in info.items():
                    if f"{{{key}}}" in answer_template:
                        question = question_template.format(disease=disease)
                        answer = answer_template.format(disease=disease, **{key: value})
                        qa_pairs.append(MedicalQAPair(question=question, answer=answer))
        
        # Expand dataset by repeating with variations
        expanded_pairs = []
        for multiplier in range(3):  # Repeat 3 times for more data
            expanded_pairs.extend(qa_pairs)
        
        return expanded_pairs
    
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


# ============================================================================
# Alternative: Simple Synthetic-Only Dataset (No External Dependencies)
# ============================================================================

class SimpleMedicalQADataset(Dataset):
    """Simple medical QA dataset using only synthetic data (no external downloads)"""
    
    def __init__(self, tokenizer, max_length: int = 512, client_id: int = None, total_clients: int = 3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.client_id = client_id
        self.total_clients = total_clients
        self.qa_pairs = self._create_medical_data()
        
    def _create_medical_data(self) -> List[MedicalQAPair]:
        """Create comprehensive synthetic medical QA data"""
        logger.info("Creating synthetic medical QA data...")
        
        # Comprehensive medical QA pairs
        qa_pairs = [
            # Diabetes
            MedicalQAPair("What are the symptoms of diabetes?", 
                         "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow-healing wounds."),
            MedicalQAPair("How is diabetes diagnosed?", 
                         "Diabetes is diagnosed through blood glucose tests, HbA1c tests, and glucose tolerance tests."),
            MedicalQAPair("What causes diabetes?", 
                         "Diabetes is caused by insufficient insulin production or insulin resistance."),
            
            # Hypertension  
            MedicalQAPair("What are the symptoms of hypertension?", 
                         "Symptoms of hypertension include headaches, shortness of breath, nosebleeds, and chest pain."),
            MedicalQAPair("How is hypertension treated?", 
                         "Hypertension is treated with lifestyle changes, antihypertensive medications, and regular monitoring."),
            
            # Heart Disease
            MedicalQAPair("What are the risk factors for heart disease?", 
                         "Risk factors for heart disease include high cholesterol, diabetes, smoking, family history, and sedentary lifestyle."),
            MedicalQAPair("How can heart disease be prevented?", 
                         "Heart disease can be prevented through healthy diet, regular exercise, smoking cessation, and stress management."),
            
            # General Medical Questions
            MedicalQAPair("What is pneumonia?", 
                         "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid."),
            MedicalQAPair("How is pneumonia treated?", 
                         "Pneumonia is treated with antibiotics, rest, fluids, and sometimes hospitalization for severe cases."),
            MedicalQAPair("What are the symptoms of migraine?", 
                         "Migraine symptoms include severe headache, nausea, vomiting, and sensitivity to light and sound."),
        ]
        
        # Multiply data for more samples
        expanded_data = []
        for _ in range(50):  # Create 500 total samples (10 * 50)
            expanded_data.extend(qa_pairs)
        
        # Distribute data among clients
        if self.client_id is not None:
            expanded_data = self._distribute_data(expanded_data)
            
        logger.info(f"Created {len(expanded_data)} medical QA samples")
        return expanded_data
        
    def _distribute_data(self, qa_pairs: List[MedicalQAPair]) -> List[MedicalQAPair]:
        """Distribute data among clients"""
        total_samples = len(qa_pairs)
        samples_per_client = total_samples // self.total_clients
        
        start_idx = self.client_id * samples_per_client
        end_idx = start_idx + samples_per_client
        
        if self.client_id == self.total_clients - 1:
            end_idx = total_samples
            
        return qa_pairs[start_idx:end_idx]
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        
        text = f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
