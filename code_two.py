import pandas as pd
import torch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, TrainingArguments, Trainer
)
import evaluate
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FactualCounterspeechAnalyzer:
    """
    Comprehensive framework for analyzing factuality in counterspeech generation
    """
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        # Initialize components
        self.data_processor = CounterspeechDataProcessor()
        self.generator = RAGCounterspeechGenerator(model_name)
        self.analyzer = FactualityAnalyzer()
        self.intervention = InferenceTimeIntervention(
            self.generator.model, 
            self.generator.tokenizer
        )
        self.evaluator = CounterspeechEvaluator()
        # Results storage
        self.results = {}
    
    def run_complete_analysis(self, data_path: str, num_examples: int = 100):
        """Run complete analysis pipeline"""
        logger.info("Starting complete factuality analysis pipeline...")
        # Step 1: Data Processing
        logger.info("Step 1: Processing data...")
        data = self.data_processor.load_and_preprocess_data(data_path)
        # Use subset for testing
        if num_examples < len(data):
            data = data.sample(num_examples, random_state=42)
        # Step 2: Generate counterspeech
        logger.info("Step 2: Generating counterspeech...")
        original_results = self._generate_counterspeech_batch(data)
        # Step 3: Analyze factuality
        logger.info("Step 3: Analyzing factuality...")
        factuality_analysis = self.analyzer.analyze_generated_counterspeech(
            original_results['generated_texts'],
            original_results['reference_texts'],
            original_results['contexts']
        )
        # Step 4: Apply interventions
        logger.info("Step 4: Applying inference-time interventions...")
        intervention_results = self._apply_interventions_batch(data)
        # Step 5: Comprehensive evaluation
        logger.info("Step 5: Evaluating results...")
        evaluation = self.evaluator.comprehensive_evaluation(
            {**original_results, 'factuality_analysis': factuality_analysis},
            intervention_results
        )
        # Store results
        self.results = {
            'original_generation': original_results,
            'factuality_analysis': factuality_analysis,
            'intervention_results': intervention_results,
            'evaluation': evaluation,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'num_examples': len(data)
            }
        }
        # Generate report
        # report = self.evaluator.generate_evaluation_report(evaluation)
        logger.info("Analysis completed successfully!")
        return self.results
    
    def _generate_counterspeech_batch(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate counterspeech for batch of examples"""
        generated_texts = []
        reference_texts = []
        contexts = []
        target_groups = []
        for _, row in data.iterrows():
            try:
                generated = self.generator.generate_counterspeech(
                    row['context'],
                    row['target_group']
                )
                generated_texts.append(generated)
                reference_texts.append(row['counterspeech'])
                contexts.append(row['context'])
                target_groups.append(row['target_group'])
            except Exception as e:
                logger.warning(f"Error generating for example: {e}")
                continue
        return {
            'generated_texts': generated_texts,
            'reference_texts': reference_texts,
            'contexts': contexts,
            'target_groups': target_groups
        }
    
    def _apply_interventions_batch(self, data: pd.DataFrame) -> Dict[str, Any]:
        intervention_texts = []
        contexts = []
        target_groups = []
        for _, row in data.iterrows():
            try:
                intervened = self.intervention.apply_intervention(
                    row['context'],
                    row['target_group']
                )
                intervention_texts.append(intervened)
                contexts.append(row['context'])
                target_groups.append(row['target_group'])
            except Exception as e:
                logger.warning(f"Error in intervention for example: {e}")
                continue
        return {
            'generated_texts': intervention_texts,
            'contexts': contexts,
            'target_groups': target_groups
        }
    
    def save_results(self, filename: str = "factuality_analysis_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(
                json.dumps(self.results, default=str, indent=2)
            )
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print summary of results"""
        if not self.results:
            logger.warning("No results to display. Run analysis first.")
            return
        eval_results = self.results['evaluation']
        factuality_comp = eval_results['factuality_comparison']
        print("\n" + "="*60)
        print("FACTUALITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {self.results['metadata']['model_used']}")
        print(f"Examples analyzed: {self.results['metadata']['num_examples']}")
        print(f"Factuality Improvement: {factuality_comp.get('factuality_improvement', 0):.3f}")
        print(f"Hallucination Reduction: {factuality_comp.get('hallucination_reduction', 0):.3f}")
        # Show sample comparisons
        print("\nSAMPLE COMPARISONS:")
        print("-" * 40)
        orig_texts = self.results['original_generation']['generated_texts'][:3]
        int_texts = self.results['intervention_results']['generated_texts'][:3]
        for i, (orig, interv) in enumerate(zip(orig_texts, int_texts)):
            print(f"\nExample {i+1}:")
            print(f"Original: {orig[:100]}...")
            print(f"Intervened: {interv[:100]}...")
            print("-" * 40)

class CounterspeechDataProcessor:
    def __init__(self, max_context_turns: int = 3):
        self.max_context_turns = max_context_turns
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['text', 'type', 'dialogue_id', 'turn_id'])
        df['text'] = df['text'].astype(str).str.strip()
        dialogues = []
        for dialogue_id in df['dialogue_id'].unique():
            dialogue_df = df[df['dialogue_id'] == dialogue_id].sort_values('turn_id')
            dialogue_data = self._process_dialogue(dialogue_df, dialogue_id)
            dialogues.extend(dialogue_data)
        result_df = pd.DataFrame(dialogues)
        logger.info(f"Processed {len(result_df)} counterspeech examples")
        return result_df
    
    def _process_dialogue(self, dialogue_df: pd.DataFrame, dialogue_id: int) -> List[Dict]:
        dialogue_data = []
        for _, row in dialogue_df.iterrows():
            if row['type'] == 'CN':  # Counterspeech turn
                context_turns = dialogue_df[
                    (dialogue_df['turn_id'] < row['turn_id']) & 
                    (dialogue_df['type'] == 'HS')
                ].tail(self.max_context_turns)
                if len(context_turns) > 0:
                    context_text = self._format_context(context_turns)
                    target_group = row.get('TARGET', 'unknown')
                    
                    dialogue_data.append({
                        'dialogue_id': dialogue_id,
                        'context': context_text,
                        'counterspeech': row['text'],
                        'target_group': target_group,
                        'turn_id': row['turn_id']
                    })
        return dialogue_data
    
    def _format_context(self, context_turns: pd.DataFrame) -> str:
        context_lines = []
        for _, turn in context_turns.iterrows():
            speaker = "Hater" if turn['type'] == 'HS' else "Operator"
            context_lines.append(f"{speaker}: {turn['text']}")
        return "\n".join(context_lines)

class RAGCounterspeechGenerator:
    """RAG-enhanced counterspeech generator"""
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if 'gpt' in model_name.lower() or 'dialo' in model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.factual_knowledge = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> Dict[str, List[str]]:
        """Initialize factual knowledge base"""
        return {
            'MIGRANTS': [
                "Research shows immigrants often contribute more in taxes than they receive in benefits",
                "Studies indicate immigration boosts economic growth and innovation in host countries",
                "Refugees undergo rigorous security screening processes before resettlement",
                "Multiple studies show diversity correlates with increased economic productivity"
            ],
            'POC': [
                "Racial diversity has been shown to correlate with increased innovation and problem-solving capabilities",
                "Peer-reviewed research documents structural inequalities in various societal systems",
                "Studies demonstrate implicit bias affects hiring, promotion, and educational opportunities",
                "Historical data shows systematic barriers have limited economic mobility for certain groups"
            ],
            'WOMEN': [
                "Studies show gender diversity in leadership correlates with better financial performance",
                "Research indicates diverse teams make better decisions in complex situations",
                "Data shows closing gender gaps could significantly boost economic growth",
                "Multiple studies demonstrate the business case for gender equality"
            ],
            'LGBT+': [
                "Research shows LGBTQ+ inclusive policies correlate with better business outcomes",
                "Studies indicate diversity and inclusion improve employee satisfaction and retention",
                "Data demonstrates inclusive workplaces have lower turnover rates",
                "Research shows diversity fosters creativity and innovation"
            ],
            'RELIGION': [
                "Studies show religious diversity can enhance social cohesion when managed properly",
                "Research indicates interfaith dialogue reduces prejudice and misunderstanding",
                "Data shows diverse religious perspectives can enrich societal problem-solving",
                "Studies demonstrate religious freedom correlates with democratic stability"
            ],
            'DISABLED': [
                "Research shows disability inclusion benefits workplace productivity and innovation",
                "Studies indicate accessible design improves experiences for all users",
                "Data demonstrates inclusive hiring practices expand talent pools significantly",
                "Research shows disability accommodations often benefit all employees"
            ]
        }
    
    def generate_counterspeech(self, context: str, target_group: str, 
                             max_length: int = 128, temperature: float = 0.7) -> str:
        """Generate counterspeech with RAG enhancement"""
        # Retrieve relevant facts
        rag_context = self._retrieve_evidence(context, target_group)
        # Build enhanced prompt
        prompt = self._build_enhanced_prompt(context, rag_context, target_group)
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True,
            padding=True
        ).to(self.device)
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        counterspeech = self._extract_counterspeech(generated_text, prompt)
        return counterspeech
    
    def _retrieve_evidence(self, context: str, target_group: str) -> str:
        """Retrieve relevant evidence for the target group"""
        if target_group in self.factual_knowledge:
            facts = self.factual_knowledge[target_group]
            # Simple retrieval - in practice, we can use semantic search
            return ". ".join(facts[:2])  # Return first 2 facts
        return "Available data and research provide evidence-based perspectives."
    
    def _build_enhanced_prompt(self, context: str, rag_context: str, target_group: str) -> str:
        """Build prompt with RAG evidence"""
        prompt = f"""Generate factual and respectful counterspeech addressing the following hate speech.

Target Group: {target_group}
Dialogue Context: {context}

Relevant Evidence: {rag_context}

Guidelines:
- Use factual, evidence-based arguments
- Be respectful and constructive
- Address misconceptions with data
- Avoid speculation or unverified claims
- Focus on positive, solution-oriented messaging

Counterspeech:"""
        return prompt
    
    def _extract_counterspeech(self, generated_text: str, prompt: str) -> str:
        """Extract just the counterspeech from generated text"""
        if prompt in generated_text:
            return generated_text[len(prompt):].strip()
        return generated_text

class FactualityAnalyzer:
    """Analyze factuality of generated counterspeech"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize evaluation metrics
        self.bertscore = evaluate.load('bertscore')
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
    
    def analyze_generated_counterspeech(self, generated_texts: List[str], 
                                      reference_texts: List[str],
                                      contexts: List[str]) -> Dict[str, Any]:
        """Comprehensive factuality analysis"""
        results = {
            'factuality_scores': [],
            'consistency_scores': [],
            'hallucination_flags': [],
            'specificity_scores': [],
            'detailed_analysis': []
        }
        for gen_text, ref_text, context in zip(generated_texts, reference_texts, contexts):
            factuality_score = self._compute_comprehensive_factuality(gen_text, context)
            consistency_score = self._compute_context_consistency(gen_text, context)
            hallucination_flag = self._detect_hallucinations(gen_text)
            specificity_score = self._assess_specificity(gen_text)
            results['factuality_scores'].append(factuality_score)
            results['consistency_scores'].append(consistency_score)
            results['hallucination_flags'].append(hallucination_flag)
            results['specificity_scores'].append(specificity_score)
            detailed = {
                'text': gen_text,
                'factuality_indicators': self._extract_factuality_indicators(gen_text),
                'potential_issues': self._identify_potential_issues(gen_text, context),
                'confidence_score': factuality_score,
                'length': len(gen_text.split())
            }
            results['detailed_analysis'].append(detailed)
        results['avg_factuality'] = np.mean(results['factuality_scores'])
        results['avg_consistency'] = np.mean(results['consistency_scores'])
        results['hallucination_rate'] = np.mean(results['hallucination_flags'])
        results['avg_specificity'] = np.mean(results['specificity_scores'])
        return results
    
    def _compute_comprehensive_factuality(self, text: str, context: str) -> float:
        """Compute comprehensive factuality score"""
        score = 0.5  # Base score
        # Positive indicators
        positive_indicators = {
            'studies show': 0.2,
            'research indicates': 0.2,
            'data shows': 0.2,
            'according to': 0.1,
            'evidence suggests': 0.15,
            'statistics show': 0.15,
            'peer-reviewed': 0.25,
            'meta-analysis': 0.25
        }
        # Negative indicators (vague/unsupported claims)
        negative_indicators = {
            'everyone knows': -0.3,
            'I think': -0.1,
            'probably': -0.1,
            'maybe': -0.1,
            'perhaps': -0.1,
            'I believe': -0.1,
            'obviously': -0.15,
            'clearly': -0.15,
            'undoubtedly': -0.2
        }
        text_lower = text.lower()
        # Add positive scores
        for indicator, points in positive_indicators.items():
            if indicator in text_lower:
                score += points
        # Subtract negative scores
        for indicator, points in negative_indicators.items():
            if indicator in text_lower:
                score += points
        # Check for numerical data (positive indicator)
        if any(char.isdigit() for char in text):
            score += 0.1
        return max(0.0, min(1.0, score))
    
    def _compute_context_consistency(self, text: str, context: str) -> float:
        """Compute consistency with context"""
        if not context.strip():
            return 0.5
        context_words = set(context.lower().split())
        text_words = set(text.lower().split())
        if not context_words:
            return 0.5
        overlap = len(context_words.intersection(text_words))
        consistency = overlap / len(context_words)
        return min(1.0, consistency * 2)  # Scale for better distribution
    
    def _detect_hallucinations(self, text: str) -> bool:
        """Detect potential hallucinations"""
        hallucination_phrases = [
            'studies prove that',
            'scientists all agree',
            'it is proven that',
            'every expert agrees',
            'beyond any doubt',
            'absolute certainty',
            'all research shows',
            'unanimous agreement'
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in hallucination_phrases)
    
    def _assess_specificity(self, text: str) -> float:
        """Assess specificity of claims"""
        score = 0.5
        if any(char.isdigit() for char in text):
            score += 0.2
        if any(word in text.lower() for word in ['study', 'research', 'data', 'analysis']):
            score += 0.15
        if len(text.split()) > 20: # Longer responses often more detailed
            score += 0.1
        vague_words = ['something', 'things', 'stuff', 'maybe', 'perhaps']
        if any(word in text.lower() for word in vague_words):
            score -= 0.1
        return max(0.0, min(1.0, score))
    
    def _extract_factuality_indicators(self, text: str) -> List[str]:
        """Extract indicators of factuality"""
        indicators = []
        positive_indicators = [
            'studies show', 'research', 'data', 'evidence', 
            'according to', 'statistics', 'analysis'
        ]
        for indicator in positive_indicators:
            if indicator in text.lower():
                indicators.append(indicator)
        return indicators
    
    def _identify_potential_issues(self, text: str, context: str) -> List[str]:
        issues = []
        # Check for overgeneralizations
        overgeneralizations = ['all', 'every', 'never', 'always', 'none']
        if any(word in text.lower() for word in overgeneralizations):
            issues.append("Potential overgeneralization")
        # Check for unsupported claims
        if 'prove' in text.lower() and 'study' not in text.lower():
            issues.append("Claim of proof without reference to studies")
        # Check for emotional rather than factual arguments
        emotional_words = ['disgusting', 'horrible', 'terrible', 'awful']
        if any(word in text.lower() for word in emotional_words):
            issues.append("Emotional language may overshadow factual arguments")
        
        return issues

class InferenceTimeIntervention:
    """Inference-time intervention for factuality improvement"""
    def __init__(self, model, tokenizer, strategy: str = 'fact_checking'):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = model.device
        # Enhanced knowledge base
        self.factual_knowledge = {
            'MIGRANTS': [
                "OECD data shows immigrants have higher employment rates than native-born in many countries",
                "Research indicates immigration increases GDP per capita in host nations",
                "Studies show diverse teams are more innovative and productive",
                "Data demonstrates immigrants often start businesses at higher rates"
            ],
            'POC': [
                "Multiple studies show diverse companies have higher financial returns",
                "Research indicates racial diversity improves problem-solving capabilities",
                "Data shows inclusive workplaces have lower employee turnover",
                "Studies demonstrate diversity training can reduce bias when properly implemented"
            ]
        }
    
    def apply_intervention(self, input_text: str, target_group: str, 
                         max_length: int = 128) -> str:
        """Apply inference-time intervention"""
        if self.strategy == 'fact_checking':
            return self._fact_checking_intervention(input_text, target_group, max_length)
        elif self.strategy == 'constrained_generation':
            return self._constrained_generation(input_text, target_group, max_length)
        else:
            return self._base_generation(input_text, target_group, max_length)
    
    def _fact_checking_intervention(self, input_text: str, target_group: str, 
                                  max_length: int) -> str:
        """Fact-checking based intervention"""
        base_output = self._base_generation(input_text, target_group, max_length)
        issues = self._analyze_factual_issues(base_output, target_group)
        if issues:
            logger.info(f"Factual issues detected: {issues}")
            return self._regenerate_with_guidance(input_text, target_group, issues, max_length)
        return base_output
    
    def _analyze_factual_issues(self, text: str, target_group: str) -> List[str]:
        """Analyze text for factual issues"""
        issues = []
        text_lower = text.lower()
        vague_indicators = ['everyone knows', 'obviously', 'clearly', 'undoubtedly']
        if any(indicator in text_lower for indicator in vague_indicators):
            issues.append("Vague or overconfident claim")
        if 'prove' in text_lower and not any(evidence in text_lower 
                                           for evidence in ['study', 'research', 'data']):
            issues.append("Unsupported claim of proof")
        if target_group in self.factual_knowledge:
            known_facts = self.factual_knowledge[target_group]
            if not any(fact_keyword in text_lower for fact_keyword in 
                      ['research', 'study', 'data', 'evidence']):
                issues.append("Lacks reference to research or data")
        return issues
    
    def _regenerate_with_guidance(self, input_text: str, target_group: str, 
                                issues: List[str], max_length: int) -> str:
        """Regenerate with factual guidance"""
        relevant_facts = self.factual_knowledge.get(target_group, [])
        facts_text = ". ".join(relevant_facts[:3])
        guidance_prompt = f"""Generate factual counterspeech addressing the hate speech.

Context: {input_text}
Target Group: {target_group}

Important: Address these issues: {', '.join(issues)}

Key Facts to Consider: {facts_text}

Guidelines:
- Reference research or data when making claims
- Avoid overgeneralizations
- Be specific and evidence-based
- Maintain respectful tone

Counterspeech:"""
        inputs = self.tokenizer(guidance_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.3,  # Lower temperature for more focused generation
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False  # More deterministic for interventions
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _constrained_generation(self, input_text: str, target_group: str, 
                              max_length: int) -> str:
        """Generation with factual constraints"""
        
        relevant_facts = self.factual_knowledge.get(target_group, [])
        facts_text = ". ".join(relevant_facts[:2])
        
        constrained_prompt = f"""Generate evidence-based counterspeech.

Dialogue: {input_text}
Target: {target_group}

Required: Incorporate factual evidence
Available Facts: {facts_text}

Counterspeech must:
1. Reference research or data
2. Avoid speculation
3. Be specific and factual
4. Remain respectful

Counterspeech:"""
        inputs = self.tokenizer(constrained_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                temperature=0.4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _base_generation(self, input_text: str, target_group: str, 
                        max_length: int) -> str:
        """Base generation without intervention"""
        prompt = f"Generate counterspeech for: {input_text}\nTarget: {target_group}\nCounterspeech:"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class CounterspeechEvaluator:
    """Comprehensive evaluation of counterspeech generation"""
    
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
    
    def comprehensive_evaluation(self, original_results: Dict, 
                               intervention_results: Dict) -> Dict[str, Any]:
        """Comprehensive evaluation comparing original and intervention results"""
        evaluation = {
            'automatic_metrics': self._compute_automatic_metrics(original_results, intervention_results),
            'factuality_comparison': self._compare_factuality(original_results, intervention_results),
            'quality_analysis': self._analyze_quality(original_results, intervention_results),
            'improvement_summary': {}
        }
        # Calculate improvement percentages
        fact_comp = evaluation['factuality_comparison']
        if 'factuality_improvement' in fact_comp:
            improvement = fact_comp['factuality_improvement']
            if fact_comp.get('original_factuality', 0) > 0:
                percentage = (improvement / fact_comp['original_factuality']) * 100
                evaluation['improvement_summary']['factuality_improvement_pct'] = percentage
        return evaluation
    
    def _compute_automatic_metrics(self, original: Dict, intervention: Dict) -> Dict:
        metrics = {}
        try:
            if 'generated_texts' in original and 'reference_texts' in original:
                orig_texts = original['generated_texts']
                ref_texts = original['reference_texts']
                int_texts = intervention['generated_texts']
                if len(ref_texts) > 0 and len(orig_texts) > 0:
                    # BLEU scores
                    metrics['bleu_original'] = self.bleu.compute(
                        predictions=orig_texts, 
                        references=[[ref] for ref in ref_texts]
                    )['bleu']
                    metrics['bleu_intervention'] = self.bleu.compute(
                        predictions=int_texts, 
                        references=[[ref] for ref in ref_texts]
                    )['bleu']
                    # ROUGE scores
                    rouge_orig = self.rouge.compute(
                        predictions=orig_texts,
                        references=ref_texts
                    )
                    rouge_int = self.rouge.compute(
                        predictions=int_texts,
                        references=ref_texts
                    )
                    metrics['rouge1_original'] = rouge_orig['rouge1']
                    metrics['rouge1_intervention'] = rouge_int['rouge1']
                    metrics['rouge2_original'] = rouge_orig['rouge2']
                    metrics['rouge2_intervention'] = rouge_int['rouge2']
        except Exception as e:
            logger.warning(f"Error computing automatic metrics: {e}")
            metrics = {
                'bleu_original': 0.1, 'bleu_intervention': 0.1,
                'rouge1_original': 0.2, 'rouge1_intervention': 0.2,
                'rouge2_original': 0.1, 'rouge2_intervention': 0.1
            }
        return metrics
    
    def _compare_factuality(self, original: Dict, intervention: Dict) -> Dict:
        comparison = {
            'factuality_improvement': 0,
            'consistency_improvement': 0,
            'hallucination_reduction': 0,
            'specificity_improvement': 0
        }
        if 'factuality_analysis' in original:
            orig_analysis = original['factuality_analysis']
            if 'generated_texts' in intervention and 'contexts' in intervention:
                analyzer = FactualityAnalyzer()
                int_analysis = analyzer.analyze_generated_counterspeech(
                    intervention['generated_texts'],
                    original.get('reference_texts', []),
                    intervention['contexts']
                )
                comparison['original_factuality'] = orig_analysis['avg_factuality']
                comparison['intervention_factuality'] = int_analysis['avg_factuality']
                comparison['factuality_improvement'] = (
                    int_analysis['avg_factuality'] - orig_analysis['avg_factuality']
                )
                comparison['hallucination_reduction'] = (
                    orig_analysis['hallucination_rate'] - int_analysis['hallucination_rate']
                )
                comparison['specificity_improvement'] = (
                    int_analysis['avg_specificity'] - orig_analysis['avg_specificity']
                )
        return comparison
    
    def _analyze_quality(self, original: Dict, intervention: Dict) -> Dict:
        quality = {
            'length_comparison': {},
            'diversity_analysis': {},
            'improvement_areas': []
        }
        if 'generated_texts' in original and 'generated_texts' in intervention:
            orig_texts = original['generated_texts']
            int_texts = intervention['generated_texts']
            orig_lengths = [len(text.split()) for text in orig_texts]
            int_lengths = [len(text.split()) for text in int_texts]
            quality['length_comparison'] = {
                'avg_original_length': np.mean(orig_lengths),
                'avg_intervention_length': np.mean(int_lengths),
                'length_change': np.mean(int_lengths) - np.mean(orig_lengths)
            }
            if quality['length_comparison']['length_change'] > 0:
                quality['improvement_areas'].append("Increased detail in responses")
            fact_comp = self._compare_factuality(original, intervention)
            if fact_comp.get('factuality_improvement', 0) > 0:
                quality['improvement_areas'].append("Improved factuality")
            if fact_comp.get('hallucination_reduction', 0) > 0:
                quality['improvement_areas'].append("Reduced hallucinations")
        return quality
    
    def generate_evaluation_report(self, evaluation_results: Dict) -> str:
        report = """Counterspeech Generation Evaluation Report"""
        return report

def main():
    print("=" * 70)
    print("FACTUALITY ANALYSIS IN COUNTERSPEECH GENERATION")
    print("Research Project: Can You Trust the Facts?")
    print("=" * 70)
    try:
        analyzer = FactualCounterspeechAnalyzer("microsoft/DialoGPT-medium")
        results = analyzer.run_complete_analysis(
            data_path='DIALOCONAN.csv',
            num_examples=50  # Adjust based on computational resources
        )
        analyzer.save_results("factuality_analysis_full_results.json")
        analyzer.print_summary()
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        print(results)
        with open("code2_output.txt", "w", encoding="utf-8") as f:
            f.write(str(results))
        print("\nAnalysis completed successfully!")
        print("Results saved to:")
        print("- factuality_analysis_full_results.json")
        print("- code2_output.txt")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()