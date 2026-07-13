import gradio as gr
import sys
import os
from pathlib import Path
import time
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_variations_2_3.inference_lora_datapters_full import load_model_and_predict, ADAPTER_REGISTRY
from classical_vs_transformer_2_1.inference_classical_roberta import load_classical_and_predict, load_fasttext_and_predict, load_roberta_and_predict, MODEL_REGISTRY, LABEL_MAPS
from encoder_variations_2_2.inference_time import predict_sarcasm_2_2

# ----------- 2.2 + 2.3 Adapter Registry to Radio Button Mapping ----------- #

LLM_UI_MAP = {
    "ENGLISH": "en-UK",
    "AUSTRALIAN": "en-AU",
    "INDIAN": "en-IN",
    "COMBINED": "combined"
}
# ----------- 2.1 Adapter Registry to Radio Button Mapping ----------- #
ROBERTA_UI_MAP = {    
    "ENGLISH": "en-UK",
    "AUSTRALIAN": "en-AU",
    "INDIAN": "en-IN",
}

COMBINED_UI_MAP = {
    "COMBINED": "combined"
}

# --------------------- 2.3 Prediction Function --------------------- #

def predict_sarcasm(arch, ui_map, language, comment):
    if not language or not comment:
        return "Error", "Please provide both a language selection and a comment for analysis."
    
    try:
        registry_key = ui_map[language]
        relative_adapter_path = ADAPTER_REGISTRY[arch][registry_key]

        # Path to the adapter files
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        base_model_path = project_root / "llm_variations_2_3"
        adapter_path = base_model_path / relative_adapter_path

        start_time = time.perf_counter()
        pred_class, prob_sarc = load_model_and_predict(arch, str(adapter_path), comment, registry_key)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        label = "Sarcastic" if pred_class == 1 else "Not Sarcastic"
        conf = prob_sarc if pred_class == 1 else (1 - prob_sarc)

        return f"{label}", f"Confidence: {conf:.1%}", f"Latency: {latency_ms:.2f} ms"
    
    except Exception as e:
        return "Error", f"Model failed to run: {str(e)}", "N/A"
    

def tl_detect_wrapper(language, comment):
    return predict_sarcasm("tinyllama", LLM_UI_MAP, language, comment)
def xlmr_detect_wrapper(language, comment):
    return predict_sarcasm("xlmr", LLM_UI_MAP, language, comment)
def qw_detect_wrapper(language, comment):
    return predict_sarcasm("qwen25", LLM_UI_MAP, language, comment)
def xlmr_lora_detect_wrapper(language, comment):
    return predict_sarcasm("xlmr_lora", LLM_UI_MAP, language, comment)

# --------------------- 2.1 Prediction Function --------------------- #
def predict_sarcasm_2_1(arch, variety_key, text, ui_map):
    if not variety_key or not text:
        return "Error", "Please provide both a variety selection and a comment for analysis."
    
    try:
        variety = ui_map[variety_key]

        script_dir = Path(__file__).resolve().parent.parent
        base_path = script_dir / "classical_vs_transformer_2_1" / "saved_models"
        
        start_time = time.perf_counter()
        combined_results ={}

        # Run Sarcasm and Sentiment Results
        for task in ["sarcasm", "sentiment"]:

            if variety in MODEL_REGISTRY[arch][task]:
                target_variety = variety
            else:
                target_variety = "combined"

            raw_dir = MODEL_REGISTRY[arch][task][target_variety]
            model_dir = base_path / raw_dir
            if not model_dir.exists():
                return "Error", f"Model directory not found: {model_dir}", "N/A"
            
            if arch == "classical":
                pred_class, confidence = load_classical_and_predict(str(model_dir), text, task, target_variety)
            elif arch == "fasttext":
                pred_class, confidence = load_fasttext_and_predict(str(model_dir), text, task, target_variety)
            elif arch == "roberta":
                pred_class, confidence = load_roberta_and_predict(str(model_dir), text, task, target_variety)
            else:
                return "Error", f"Unknown architecture: {arch}", "N/A"
            
            labels = LABEL_MAPS[task]
            combined_results[task] = (labels[pred_class], confidence)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Combined Results
        sarcasm_label, sarcasm_conf = combined_results["sarcasm"]
        sentiment_label, sentiment_conf = combined_results["sentiment"]


        return (
            f"{sarcasm_label}  |  {sentiment_label}",
            f"Sarcasm: {sarcasm_conf:.1%}  |  Sentiment: {sentiment_conf:.1%}",
            f"Latency: {latency_ms:.2f} ms"
        )
    except Exception as e:
        return "Error", f"Model failed to run: {str(e)}","N/A"
    
    
# 2.1 wrapper functions
def classical_wrapper(variety, text):
    return predict_sarcasm_2_1("classical", variety, text, COMBINED_UI_MAP)
def fasttext_wrapper(variety, text):
    return predict_sarcasm_2_1("fasttext", variety, text, COMBINED_UI_MAP)
def roberta_wrapper(variety, text):
    return predict_sarcasm_2_1("roberta", variety, text, ROBERTA_UI_MAP)

# 2.2 wrapper functions
def deberta_wrapper(variety, text):
    return predict_sarcasm_2_2("deberta", variety, text, LLM_UI_MAP)
def rembert_wrapper(variety, text):
    return predict_sarcasm_2_2("rembert", variety, text, LLM_UI_MAP)
def xlm_roberta_wrapper(variety, text):
    return predict_sarcasm_2_2("xlm_roberta", variety, text, LLM_UI_MAP)

# ------------------ Flagging ------------------ #
BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR / "flagged_results"
DEPLOYMENT_DIR.mkdir(exist_ok=True)

def log_flag(model, language, comment, result, conf):
    filename = f"flagged_{model}_results.csv"
    filepath = DEPLOYMENT_DIR / filename
    file_exists = filepath.exists()

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["MODEL", "LANGUAGE", "COMMENT", "RESULT", "CONFIDENCE"])

        writer.writerow([model, language, comment, result, conf])

def handle_flag(model, language, comment, result, conf):
    if not comment or not result:
        return
    log_flag(model, language, comment, result, conf)

# 2.3 flag wrappers
def tl_flag_wrapper(lang, txt, res, conf):
    return handle_flag("tinyllama", lang, txt, res, conf)
def xlmr_flag_wrapper(lang, txt, res, conf):
    return handle_flag("xlmr", lang, txt, res, conf)
def qw_flag_wrapper(lang, txt, res, conf):
    return handle_flag("qwen25", lang, txt, res, conf)
def xlmr_lora_flag_wrapper(lang, txt, res, conf):
    return handle_flag("xlmr_lora", lang, txt, res, conf)

# 2.1 flag wrappers
def classical_flag_wrapper(lang, txt, res, conf):
    return handle_flag("classical_sentmient", lang, txt, res, conf)
def fasttext_flag_wrapper(lang, txt, res, conf):
    return handle_flag("fasttext_sentmient", lang, txt, res, conf)
def roberta_flag_wrapper(lang, txt, res, conf):
    return handle_flag("roberta_sentiment", lang, txt, res, conf)


# 2.2 flag wrappers
def deberta_flag_wrapper(lang, txt, res, conf):
    return handle_flag("deberta", lang, txt, res, conf)
def rembert_flag_wrapper(lang, txt, res, conf):
    return handle_flag("rembert", lang, txt, res, conf)
def xlm_roberta_flag_wrapper(lang, txt, res, conf):
    return handle_flag("xlm_roberta", lang, txt, res, conf)


# ------------------ Gradio UI Layout ------------------ #

def launch_ui():
    with gr.Blocks(title="Sarcasm Detection", css="footer {visibility: hidden;}") as Sarcasm_Detection_System:
        gr.Markdown("# Sarcasm Detection System")                                           
        
        with gr.Tabs():
            # ---------------------- 2.1 Tabs ---------------------- #
            with gr.Tab("2.1 - Baseline Models"):
                with gr.Tabs():
                    # TAB 1: Classical Tab                                                                                                     
                    with gr.Tab("TF-IDF (Classical)"):
                        cl_sent_lang = gr.Radio(list(COMBINED_UI_MAP.keys()), label="Select Variety", value="COMBINED")
                        cl_sent_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")                 
                        cl_btn = gr.Button("Detect with TF-IDF")
                        cl_sent_output = [gr.Textbox(label="Result: (Sarcasm Model  |  Sentiment Model):"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        cl_sent_flag = gr.Button("Flag Result")

                        cl_btn.click(fn=classical_wrapper, inputs=[cl_sent_lang, cl_sent_text], outputs=cl_sent_output)
                                
                        cl_sent_flag.click(
                            fn=classical_flag_wrapper, 
                            inputs=[cl_sent_lang, cl_sent_text, cl_sent_output[0], cl_sent_output[1]], 
                            outputs=[]
                        )

                    # TAB 2: FastText Tab
                    with gr.Tab("FastText"):
                        ft_sent_lang = gr.Radio(list(COMBINED_UI_MAP.keys()), label="Select Variety", value="COMBINED")
                        ft_sent_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")                 
                        ft_sent_btn = gr.Button("Detect with FastText")
                        ft_sent_output = [gr.Textbox(label="Result: (Sarcasm Model  |  Sentiment Model)"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        ft_sent_flag = gr.Button("Flag Result")

                        ft_sent_btn.click(fn = fasttext_wrapper, inputs = [ft_sent_lang, ft_sent_text], outputs= ft_sent_output)
                        ft_sent_flag.click(
                            fn = fasttext_flag_wrapper, 
                            inputs = [ft_sent_lang, ft_sent_text, ft_sent_output[0], ft_sent_output[1]], 
                            outputs= []
                        )

                    # TAB 3: RoBERTa Tab
                    with gr.Tab("RoBERTa"):
                        rb_sent_lang = gr.Radio(list(ROBERTA_UI_MAP.keys()), label="Select Variety", value="ENGLISH")
                        rb_sent_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")      
                        rb_sent_btn = gr.Button("Detect with RoBERTa")
                        rb_sent_output = [gr.Textbox(label="Result: (Sarcasm Model  |  Sentiment Model)"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        rb_sent_flag = gr.Button("Flag Result")

                        rb_sent_btn.click(fn = roberta_wrapper, inputs = [rb_sent_lang, rb_sent_text], outputs= rb_sent_output)
                        rb_sent_flag.click(
                            fn = roberta_flag_wrapper,
                            inputs = [rb_sent_lang, rb_sent_text, rb_sent_output[0], rb_sent_output[1]], 
                            outputs= []
                        )
            # ---------------------- 2.2 Tabs ---------------------- #
            with gr.Tab("2.2 - Cross Variety Models"):
                with gr.Tabs():
                    # TAB 1: Classical Tab                                                                                                     
                    with gr.Tab("DeBERTa"):
                        db_sent_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Variety", value="ENGLISH")
                        db_sent_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")         
                        db_btn = gr.Button("Detect with DeBERTA")
                        db_sent_output = [gr.Textbox(label="Result: (Sarcasm Model  |  Sentiment Model):"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        db_sent_flag = gr.Button("Flag Result")

                        db_btn.click(fn=deberta_wrapper, inputs=[db_sent_lang, db_sent_text], outputs=db_sent_output)
                                
                        db_sent_flag.click(
                            fn=deberta_flag_wrapper, 
                            inputs=[db_sent_lang, db_sent_text, db_sent_output[0], db_sent_output[1]], 
                            outputs=[]
                        )

                    # TAB 2: FastText Tab
                    with gr.Tab("RemBERT"):
                        rm_sent_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Variety", value="ENGLISH")
                        rm_sent_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")  
                        rm_sent_btn = gr.Button("Detect with RemBERT")
                        rm_sent_output = [gr.Textbox(label="Result: (Sarcasm Model  |  Sentiment Model)"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        rm_sent_flag = gr.Button("Flag Result")

                        rm_sent_btn.click(fn = rembert_wrapper, inputs = [rm_sent_lang, rm_sent_text], outputs= rm_sent_output)
                        rm_sent_flag.click(
                            fn = rembert_flag_wrapper, 
                            inputs = [rm_sent_lang, rm_sent_text, rm_sent_output[0], rm_sent_output[1]], 
                            outputs= []
                        )

                    # TAB 3: XLM-RoBERTa Tab
                    with gr.Tab("XLM-RoBERTa"):
                        xr_sent_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Variety", value="ENGLISH")
                        xr_sent_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")   
                        xr_sent_btn = gr.Button("Detect with XLM-RoBERTa")
                        xr_sent_output = [gr.Textbox(label="Result:"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        xr_sent_flag = gr.Button("Flag Result")

                        xr_sent_btn.click(fn = xlm_roberta_wrapper, inputs = [xr_sent_lang, xr_sent_text], outputs= xr_sent_output)
                        xr_sent_flag.click(
                            fn = xlm_roberta_flag_wrapper,
                            inputs = [xr_sent_lang, xr_sent_text, xr_sent_output[0], xr_sent_output[1]], 
                            outputs= []
                        )
            # ---------------------- 2.3 Tabs ---------------------- #
            with gr.Tab("2.3 - LLM Variation Models"):
                with gr.Tabs():
                    # TAB 1: TinyLlama
                    with gr.Tab("TinyLlama"):
                        tl_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Language Option", value="ENGLISH")
                        tl_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")
                        tl_detect_btn = gr.Button("Detect with TinyLlama")
                        tl_result_output = [gr.Textbox(label="Result:"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        tl_flag = gr.Button("Flag Result")
                        
                        tl_detect_btn.click(
                            fn=tl_detect_wrapper,
                            inputs=[tl_lang, tl_text], outputs=tl_result_output
                        )
                        tl_flag.click(
                            fn=tl_flag_wrapper,
                            inputs=[tl_lang, tl_text, tl_result_output[0], tl_result_output[1]],                           
                            outputs=[]
                        )

                    # TAB 2: XLMR
                    with gr.Tab("XLMR"):
                        xlmr_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Language Option", value="ENGLISH")
                        xlmr_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")
                        xlmr_detect_btn = gr.Button("Detect with XLMR")
                        xlmr_result_output = [gr.Textbox(label="Result:"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                        
                        xlmr_flag = gr.Button("Flag Result")
                        xlmr_detect_btn.click(
                            fn=xlmr_detect_wrapper,
                            inputs=[xlmr_lang, xlmr_text], outputs=xlmr_result_output
                        )
                        xlmr_flag.click(
                            fn=xlmr_flag_wrapper,
                            inputs=[xlmr_lang, xlmr_text, xlmr_result_output[0], xlmr_result_output[1]],
                            outputs=[]
                        )

                    # TAB 3: Qwen2.5
                    with gr.Tab("Qwen2.5"):
                        qw_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Language Option", value="ENGLISH")
                        qw_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")
                        qw_detect_btn = gr.Button("Detect with Qwen2.5")
                        qw_result_output = [gr.Textbox(label="Result:"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                                    
                        qw_flag = gr.Button("Flag Result")
                        qw_detect_btn.click(
                            fn=qw_detect_wrapper,
                            inputs=[qw_lang, qw_text], outputs=qw_result_output
                        )
                        qw_flag.click(
                            fn=qw_flag_wrapper,
                            inputs=[qw_lang, qw_text, qw_result_output[0], qw_result_output[1]],
                            outputs=[]
                        )
                    # TAB 4: XLMR + LoRA
                    with gr.Tab("XLMR + LoRA"):
                        xlmr_lora_lang = gr.Radio(list(LLM_UI_MAP.keys()), label="Select Language Option", value="ENGLISH")
                        xlmr_lora_text = gr.Textbox(label="Enter your comment:", placeholder="e.g. Oh brilliant, it's raining again.")
                        xlmr_lora_detect_btn = gr.Button("Detect with XLMR + LoRA")
                        xlmr_lora_result_output = [gr.Textbox(label="Result:"), gr.Textbox(label="Model Confidence:"), gr.Textbox(label="Inference Latency:")]
                                    
                        xlmr_lora_flag = gr.Button("Flag Result")
                        xlmr_lora_detect_btn.click(
                            fn=xlmr_lora_detect_wrapper,
                            inputs=[xlmr_lora_lang, xlmr_lora_text], outputs=xlmr_lora_result_output
                        )
                        xlmr_lora_flag.click(
                            fn=xlmr_lora_flag_wrapper,
                            inputs=[xlmr_lora_lang, xlmr_lora_text, xlmr_lora_result_output[0], xlmr_lora_result_output[1]],
                            outputs=[]
                        )
    return Sarcasm_Detection_System.launch(share=True)

if __name__ == "__main__":
    launch_ui()
