# In-Context Learning Benchmarking: Text-to-Text and Vision Models

## Project Overview

This project implements a comprehensive benchmarking framework for evaluating **in-context learning (ICL)** capabilities across multiple transformer-based models. In-context learning refers to the ability of language models to perform new tasks by conditioning on examples provided in the input prompt, without requiring fine-tuning or gradient updates.

The project specifically evaluates how different models perform across three fundamental learning paradigms:
- **Zero-Shot Learning**: No examples provided in the context
- **One-Shot Learning**: Single example provided in the context  
- **Few-Shot Learning**: Multiple examples provided in the context

## Research Context: SIVO (Structured In-Context Vector Optimization)

This benchmarking study is grounded in the SIVO framework, which focuses on understanding and optimizing how models utilize contextual information during inference. SIVO provides insights into:

1. **Context Utilization**: How effectively models incorporate in-context examples into their predictions
2. **Prompt Structure Sensitivity**: How model performance varies with different prompt formulations
3. **Model Scaling Effects**: How in-context learning capabilities improve with model size
4. **Knowledge Transfer**: How knowledge learned during pre-training transfers to new tasks via prompting

## Models Evaluated

### 1. Flan-T5 (google/flan-t5-base)
- **Architecture**: Encoder-Decoder (Sequence-to-Sequence)
- **Parameters**: Base variant (~250M)
- **Characteristics**: 
  - Instruction-tuned model
  - Excellent at following natural language instructions
  - Strong zero-shot and few-shot performance
  - Deterministic task understanding

**Performance Summary**:
- ✅ **Zero-Shot**: Accurate capital identification
- ✅ **One-Shot**: Excellent with single example
- ✅ **Few-Shot**: Robust with multiple examples
- **Strengths**: Instruction following, task comprehension
- **Weaknesses**: None significant for this task

---

### 2. OPT (facebook/opt-125m)
- **Architecture**: Decoder-Only (Causal Language Model)
- **Parameters**: 125M
- **Characteristics**:
  - Smaller, more computationally efficient model
  - Autoregressive generation
  - Less instruction-tuned than Flan-T5
  - Trained on diverse text corpora

**Performance Summary**:
- ⚠️ **Zero-Shot**: Generates reasonable output
- ⚠️ **One-Shot**: Struggles with prompt structure
- ❌ **Few-Shot**: Generates additional questions instead of answers
- **Strengths**: Efficient inference, baseline model
- **Weaknesses**: Poor prompt comprehension, generates off-task content

---

### 3. GPT-2 (openai/gpt2)
- **Architecture**: Decoder-Only (Causal Language Model)
- **Parameters**: 124M
- **Characteristics**:
  - Standard causal language model
  - Pre-trained on WebText corpus
  - Minimal task-specific fine-tuning
  - Strong foundation for language generation

**Performance Summary**:
- ❌ **Zero-Shot**: Repeats prompt patterns
- ❌ **One-Shot**: Echoes examples without answering
- ❌ **Few-Shot**: Generates new questions instead of answers
- **Strengths**: Stable generation, good diversity
- **Weaknesses**: Severe prompt-following limitations, task confusion

**Improvement Attempt**: Adding padding token configuration shows minimal improvement in understanding prompt structure.

---

### 4. GPT-3 (TurkuNLP/gpt3-finnish-large)
- **Architecture**: Decoder-Only (Causal Language Model)
- **Parameters**: Large variant of GPT-3 architecture
- **Characteristics**:
  - Language-specific (Finnish) pre-training
  - Superior prompt understanding
  - Better in-context learning than smaller models
  - More aligned output structure

**Performance Summary**:
- ✅ **Zero-Shot**: Reasonable responses
- ✅ **One-Shot**: Good performance with example
- ✅ **Few-Shot**: Superior to GPT-2 and OPT, correctly answers questions
- **Strengths**: Excellent in-context learning, task understanding
- **Weaknesses**: Occasional irrelevant generation with increased context

---

## Dataset: CoQA (Conversational Question Answering)

The benchmark uses the **CoQA dataset** from Hugging Face:
- Multi-turn conversational question answering
- Rich contextual understanding required
- Diverse question types and answer formats
- Ideal for evaluating transfer learning capabilities

## Experimental Design

### Task Definition
The primary task is **factual question answering about world capitals**:
```
Question: What is the capital of [Country]?
Answer: [Capital_City]
```

### Evaluation Methodology

#### Zero-Shot Evaluation
```
Input: "Question: What is the capital of France? Answer:"
Expected: Model generates answer without examples
```

#### One-Shot Evaluation
```
Input: 
"Question: What is the capital of Germany?
Answer: Berlin

Question: What is the capital of France?
Answer:"
Expected: Model uses example to understand pattern
```

#### Few-Shot Evaluation
```
Input:
"Question: What is the capital of Germany?
Answer: Berlin

Question: What is the capital of Italy?
Answer: Rome

[... additional examples ...]

Question: What is the capital of Algeria?
Answer:"
Expected: Model uses multiple examples for better performance
```

### Response Extraction

For causal language models, a response extraction mechanism was implemented to isolate the actual answer from the generated text:
```python
def extract_generated_response(full_response, question):
    question_index = full_response.find(question)
    generated_response = full_response[question_index + len(question):].strip()
    return generated_response
```

## Key Findings

### Model Scaling and In-Context Learning

The experiments demonstrate a clear correlation between model size/sophistication and in-context learning performance:

| Model | Size | Architecture | Zero-Shot | One-Shot | Few-Shot |
|-------|------|--------------|-----------|----------|----------|
| Flan-T5 | 250M | Seq2Seq | ✅ Strong | ✅ Strong | ✅ Strong |
| GPT-3 Variant | Large | Decoder | ✅ Good | ✅ Good | ✅ Good |
| OPT | 125M | Decoder | ⚠️ Weak | ❌ Poor | ❌ Poor |
| GPT-2 | 124M | Decoder | ❌ Poor | ❌ Poor | ❌ Poor |

### Architecture Matters

**Encoder-Decoder (Flan-T5)**: 
- Superior task understanding
- Better prompt structure comprehension
- More reliable few-shot learning
- Natural alignment with instruction-following

**Decoder-Only (GPT-2, OPT, GPT-3)**:
- More variable performance based on scale
- Struggles with prompt structure in smaller models
- Improves significantly with model size
- Requires careful prompt engineering

### Instruction Tuning Impact

Models fine-tuned on instruction-following tasks (Flan-T5) show significantly better in-context learning performance compared to general pre-trained models. This suggests that instruction tuning provides valuable inductive biases for interpreting prompts.

### Context Utilization Patterns

1. **Prompt Structure Recognition**: Larger models better recognize Q&A patterns
2. **Example Relevance**: Models with instruction tuning leverage examples more effectively
3. **Generalization**: Few-shot learning provides clearer patterns, improving accuracy
4. **Hallucination Risk**: Smaller models may generate off-task content when confused

## Insights for SIVO Framework

This benchmarking study provides empirical evidence for SIVO principles:

### 1. **Structured Context Processing**
- Models with explicit structure recognition (Flan-T5) outperform those relying on pattern matching
- Structured prompts (clear Q&A format) improve in-context learning across all models

### 2. **Vector Space Optimization**
- Larger models create better-aligned vector representations for in-context learning
- Few-shot examples provide "anchors" that improve representation alignment

### 3. **Model Capacity Requirements**
- Minimum capacity threshold needed for reliable in-context learning
- GPT-2/OPT show this threshold behavior
- Instruction tuning reduces required capacity

### 4. **Prompt Engineering Sensitivity**
- Small variations in prompt structure significantly impact performance
- Consistent formatting crucial for causal language models
- Encoder-decoder models more robust to formatting variations

## Implementation Details

### Dependencies
```
transformers >= 4.30.0
torch >= 1.9.0
datasets
scikit-learn
```

### Key Functions

#### Response Generation
```python
def generate_response(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

#### Evaluation Metrics
- **Accuracy**: Exact match with expected capital
- **F1 Score**: Partial match scoring
- **Response Validity**: Whether model provided coherent answer

## Recommendations for Practitioners

### Model Selection

**For Production In-Context Learning:**
1. **Preferred**: Use instruction-tuned models (Flan-T5 or larger GPT variants)
2. **If Constrained**: Flan-T5 offers best size/performance trade-off
3. **Avoid**: Using small causal models without task-specific fine-tuning

### Prompt Engineering Best Practices

1. **Use Clear Structure**: Q&A format with explicit markers
2. **Include Diverse Examples**: 3-5 examples covering range of inputs
3. **Consistent Formatting**: Maintain same structure throughout examples
4. **Relevant Examples**: Choose examples similar to target task
5. **Temperature Tuning**: Lower temperature (0.1-0.3) for factual tasks

### Context Window Management

- **For Few-Shot**: 3-5 examples typically sufficient
- **Diminishing Returns**: More examples don't always improve performance
- **Token Budget**: Consider model's max token limit
- **Relevance > Quantity**: Quality of examples more important than quantity

## Future Work

1. **Multimodal Extension**: Evaluate vision-language models (CLIP, BLIP)
2. **Multilingual Evaluation**: Test cross-lingual in-context learning
3. **Task Diversity**: Extend beyond factual QA to reasoning, summarization
4. **Chain-of-Thought Prompting**: Evaluate reasoning performance
5. **Prompt Optimization**: Automated prompt engineering techniques

## References

### SIVO Framework
The SIVO (Structured In-Context Vector Optimization) framework provides theoretical grounding for understanding how models leverage contextual information. See `SIVO.pdf` for detailed theoretical framework.

### Related Work
- Brown, T. M., et al. (2020). Language Models are Few-Shot Learners. ArXiv:2005.14165
- Wei, J., et al. (2022). Emergent Abilities of Large Language Models. ArXiv:2206.07682
- Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR

## Project Structure

```
Multimodal-Context-Learning-Benchmarking/
├── README.md                                           # This file
├── SIVO.pdf                                           # Theoretical framework
├── In_Context_Learning_Benchmarking_TextToText&Vision_Models.ipynb
│   ├── Step 1: Library Installation
│   ├── Step 2: Dataset Loading
│   ├── Section 1: Flan-T5 Testing
│   │   ├── Step 3: Model Configuration
│   │   ├── Step 4: Zero-Shot Evaluation
│   │   ├── Step 5: One-Shot Evaluation
│   │   ├── Step 6: Few-Shot Evaluation
│   │   └── Step 7: Performance Analysis
│   ├── Section 2: OPT Model Testing
│   ├── Section 3: GPT-2 Model Testing
│   ├── Section 4: GPT-3 Model Testing (Finnish)
│   └── Comprehensive Comparison
└── [Future]: Vision model experiments
```

## Contributing

This is a personal research project. For questions or suggestions, please refer to the SIVO.pdf for theoretical context and methodology.

## License

All code in this project follows the same license as the original repositories of the models used. Refer to individual model licenses:
- Flan-T5: Apache 2.0 (Hugging Face)
- GPT-2: Modified MIT (OpenAI)
- OPT: OPT-175B License (Meta)

## Contact & Acknowledgments

**Project Lead**: Manel Alawad

**Acknowledgments**:
- Google Research for Flan-T5
- Meta AI for OPT
- OpenAI for GPT-2
- Hugging Face for model hosting and transformers library

---

**Last Updated**: November 2025

**Status**: Active Research - Results may evolve with additional experiments

For detailed implementation notes and experimental logs, refer to the Jupyter notebook included in this repository.
