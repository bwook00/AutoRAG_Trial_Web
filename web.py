import os
import pathlib

import gradio as gr
import pandas as pd
import yaml

from autorag.evaluator import Evaluator

from src.runner import GradioStreamRunner

root_dir = os.path.dirname(os.path.realpath(__file__))

# Paths to example files
config_dir = os.path.join(root_dir, "config")

# Non-GPU Examples
non_gpu = os.path.join(config_dir, "non_gpu")
simple_openai = os.path.join(non_gpu, "simple_openai.yaml")
simple_openai_korean = os.path.join(non_gpu, "simple_openai_korean.yaml")
compact_openai = os.path.join(non_gpu, "compact_openai.yaml")
compact_openai_korean = os.path.join(non_gpu, "compact_openai_korean.yaml")
half_openai = os.path.join(non_gpu, "half_openai.yaml")
half_openai_korean = os.path.join(non_gpu, "half_openai_korean.yaml")
full_openai = os.path.join(non_gpu, "full_no_rerank_openai.yaml")

non_gpu_examples_list = [
    simple_openai, simple_openai_korean, compact_openai, compact_openai_korean, half_openai, half_openai_korean,
    full_openai
]
non_gpu_examples = list(map(lambda x: [x], non_gpu_examples_list))

# GPU Examples
gpu = os.path.join(config_dir, "gpu")
compact_openai_gpu = os.path.join(gpu, "compact_openai.yaml")
compact_openai_korean_gpu = os.path.join(gpu, "compact_openai_korean.yaml")
half_openai_gpu = os.path.join(gpu, "half_openai.yaml")
half_openai_korean_gpu = os.path.join(gpu, "half_openai_korean.yaml")
full_openai_gpu = os.path.join(gpu, "full_no_rerank_openai.yaml")

gpu_examples_list = [
    compact_openai_gpu, compact_openai_korean_gpu, half_openai_gpu, half_openai_korean_gpu, full_openai_gpu
]
gpu_examples = list(map(lambda x: [x], gpu_examples_list))

# GPU + API
gpu_api = os.path.join(config_dir, "gpu_api")
compact_openai_gpu_api = os.path.join(gpu_api, "compact_openai.yaml")
compact_openai_korean_gpu_api = os.path.join(gpu_api, "compact_openai_korean.yaml")
half_openai_gpu_api = os.path.join(gpu_api, "half_openai.yaml")
half_openai_korean_gpu_api = os.path.join(gpu_api, "half_openai_korean.yaml")
full_openai_gpu_api = os.path.join(gpu_api, "full_no_rerank_openai.yaml")

gpu_api_examples_list = [
    compact_openai_gpu_api, compact_openai_korean_gpu_api, half_openai_gpu_api, half_openai_korean_gpu_api,
    full_openai_gpu_api
]
gpu_api_examples = list(map(lambda x: [x], gpu_api_examples_list))

example_qa_parquet = os.path.join(root_dir, "sample_data", "qa_data_sample.parquet")
example_corpus_parquet = os.path.join(root_dir, "sample_data", "corpus_data_sample.parquet")


def display_yaml(file):
    if file is None:
        return "No file uploaded"
    with open(file.name, "r") as f:
        content = yaml.safe_load(f)
    return yaml.dump(content, default_flow_style=False)


def display_parquet(file):
    if file is None:
        return pd.DataFrame()
    df = pd.read_parquet(file.name)
    return df


def check_files(yaml_file, qa_file, corpus_file):
    if yaml_file is not None and qa_file is not None and corpus_file is not None:
        return gr.update(visible=True)
    return gr.update(visible=False)


def run_trial(file, yaml_file, qa_file, corpus_file):
    project_dir = os.path.join(pathlib.PurePath(file.name).parent, "project")
    evaluator = Evaluator(qa_file, corpus_file, project_dir=project_dir)

    evaluator.start_trial(yaml_file, skip_validation=True)
    return ("❗Trial Completed❗ "
            "Go to Chat Tab to start the conversation")


def set_environment_variable(api_name, api_key):
    if api_name and api_key:
        try:
            os.environ[api_name] = api_key
            return "✅ Setting Complete ✅"
        except Exception as e:
            return f"Error setting environment variable: {e}"
    return "API Name or Key is missing"


def stream_default(file, history):
    # Default YAML Runner
    yaml_path = os.path.join(config_dir, "extracted_sample.yaml")
    project_dir = os.path.join(
        pathlib.PurePath(file.name).parent, "project"
    )
    default_gradio_runner = GradioStreamRunner.from_yaml(yaml_path, project_dir)

    history.append({"role": "assistant", "content": ""})
    # Stream responses for the chatbox
    for default_output in default_gradio_runner.stream_run(history[-2]["content"]):
        stream_delta = default_output[0]
        history[-1]["content"] = stream_delta
        yield history


def stream_optimized(file, history):
    # Custom YAML Runner
    trial_dir = os.path.join(pathlib.PurePath(file.name).parent, "project", "0")
    custom_gradio_runner = GradioStreamRunner.from_trial_folder(trial_dir)

    history.append({"role": "assistant", "content": ""})
    for output in custom_gradio_runner.stream_run(history[-2]["content"]):
        stream_delta = output[0]
        history[-1]["content"] = stream_delta
        yield history


def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}]


with gr.Blocks(theme="earneleh/paris") as demo:
    gr.Markdown("# AutoRAG Trial & Debugging Interface")

    with gr.Tabs() as tabs:
        with gr.Tab("Environment Variables"):
            gr.Markdown("## Environment Variables")
            with gr.Row():  # Arrange horizontally
                with gr.Column(scale=3):
                    api_name = gr.Textbox(
                        label="Environment Variable Name",
                        type="text",
                        placeholder="Enter your Environment Variable Name",
                    )
                    gr.Examples(examples=[["OPENAI_API_KEY"]], inputs=api_name)
                with gr.Column(scale=7):
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Enter your API Key",
                    )

            set_env_button = gr.Button("Set Environment Variable")
            env_output = gr.Textbox(
                label="Status", interactive=False
            )

            api_key.submit(
                set_environment_variable, inputs=[api_name, api_key], outputs=env_output
            )
            set_env_button.click(
                set_environment_variable, inputs=[api_name, api_key], outputs=env_output
            )

        with gr.Tab("File Upload"):
            with gr.Row() as file_upload_row:
                with gr.Column(scale=3):
                    yaml_file = gr.File(
                        label="Upload YAML File",
                        file_count="single",
                    )
                    make_yaml_button = gr.Button("Make Your Own YAML File",
                                                 link="https://tally.so/r/mBQY5N")

                with gr.Column(scale=7):
                    yaml_content = gr.Textbox(label="YAML File Content")
                    gr.Markdown("Here is the Sample YAML File. Just click the file ❗")

                    gr.Markdown("### Non-GPU Examples")
                    gr.Examples(examples=non_gpu_examples, inputs=yaml_file)

                    with gr.Row():
                        # Section for GPU examples
                        with gr.Column():
                            gr.Markdown("### GPU Examples")
                            gr.Markdown(
                                "**⚠️ Warning**: Here are the YAML files containing the modules that use the **local model**.")
                            gr.Markdown(
                                "Note that if you Run_Trial in a non-GPU environment, **it can take a very long time**.")
                            gr.Examples(examples=gpu_examples, inputs=yaml_file)
                            make_gpu = gr.Button("Use AutoRAG GPU Feature",
                                                 link="https://tally.so/r/3j7rP6")

                        # Section for GPU + API examples
                        with gr.Column():
                            gr.Markdown("### GPU + API Examples")
                            gr.Markdown(
                                "**⚠️ Warning**: Here are the YAML files containing the modules that use the **local model** and **API Based Model**.")
                            gr.Markdown("You need to set **JINA_API_KEY**, **COHERE_API_KEY**, **MXBAI_API_KEY** and **VOYAGE_API_KEY** as environment variables to use this feature. ")
                            gr.Examples(examples=gpu_api_examples, inputs=yaml_file)
                            gpu_api_button = gr.Button("Use AutoRAG API KEY Feature",
                                                       link="https://tally.so/r/waD1Ab")



            with gr.Row() as qa_upload_row:
                with gr.Column(scale=3):
                    qa_file = gr.File(
                        label="Upload qa.parquet File",
                        file_count="single",
                    )
                    # Add button for QA
                    make_qa_button = gr.Button("Make Your Own QA Data",
                                               link="https://huggingface.co/spaces/AutoRAG/AutoRAG-data-creation")

                with gr.Column(scale=7):
                    qa_content = gr.Dataframe(label="QA Parquet File Content")
                    gr.Markdown("Here is the Sample QA File. Just click the file ❗")
                    gr.Examples(examples=[[example_qa_parquet]], inputs=qa_file)
            with gr.Row() as corpus_upload_row:
                with gr.Column(scale=3):
                    corpus_file = gr.File(
                        label="Upload corpus.parquet File",
                        file_count="single",
                    )
                    make_corpus_button = gr.Button("Make Your Own Corpus Data",
                                                   link="https://huggingface.co/spaces/AutoRAG/AutoRAG-data-creation")
                with gr.Column(scale=7):
                    corpus_content = gr.Dataframe(label="Corpus Parquet File Content")
                    gr.Markdown(
                        "Here is the Sample Corpus File. Just click the file ❗"
                    )
                    gr.Examples(examples=[[example_corpus_parquet]], inputs=corpus_file)

            run_trial_button = gr.Button("Run Trial", visible=False)
            trial_output = gr.Textbox(label="Trial Output", visible=False)

            yaml_file.change(display_yaml, inputs=yaml_file, outputs=yaml_content)
            qa_file.change(display_parquet, inputs=qa_file, outputs=qa_content)
            corpus_file.change(
                display_parquet, inputs=corpus_file, outputs=corpus_content
            )

            yaml_file.change(
                check_files,
                inputs=[yaml_file, qa_file, corpus_file],
                outputs=run_trial_button,
            )
            qa_file.change(
                check_files,
                inputs=[yaml_file, qa_file, corpus_file],
                outputs=run_trial_button,
            )
            corpus_file.change(
                check_files,
                inputs=[yaml_file, qa_file, corpus_file],
                outputs=run_trial_button,
            )

            run_trial_button.click(
                lambda: (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                ),
                outputs=[
                    file_upload_row,
                    qa_upload_row,
                    corpus_upload_row,
                    trial_output,
                ],
            )
            run_trial_button.click(
                run_trial,
                inputs=[yaml_file, yaml_file, qa_file, corpus_file],
                outputs=trial_output,
            )

        # New Chat Tab
        with gr.Tab("Chat") as chat_tab:
            gr.Markdown("### Compare Chat Models")

            question_input = gr.Textbox(
                label="Your Question", placeholder="Type your question here..."
            )
            pseudo_input = gr.Textbox(label="havertz", visible=False)

            with gr.Row():
                # Left Chatbox (Default YAML)
                with gr.Column():
                    gr.Markdown("#### Naive RAG Chat")
                    default_chatbox = gr.Chatbot(label="Naive RAG Conversation",type="messages")

                # Right Chatbox (Custom YAML)
                with gr.Column():
                    gr.Markdown("#### Optimized RAG Chat")
                    custom_chatbox = gr.Chatbot(label="Optimized RAG Conversation",type="messages")

            question_input.submit(lambda x: x, inputs=[question_input], outputs=[pseudo_input]).then(
                user, [question_input, default_chatbox], outputs=[question_input, default_chatbox], queue=False
            ).then(
                stream_default,
                inputs=[yaml_file, default_chatbox],
                outputs=[default_chatbox],
            )

            pseudo_input.change(
                user, [pseudo_input, custom_chatbox], outputs=[question_input, custom_chatbox], queue=False).then(
                stream_optimized,
                inputs=[yaml_file, custom_chatbox],
                outputs=[custom_chatbox],
            )


            deploy_button = gr.Button("Deploy",
                                       link="https://tally.so/r/3XM7y4")


if __name__ == "__main__":
    # Run the interface
    demo.launch(share=False, debug=True)
