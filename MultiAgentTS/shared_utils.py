import time
import os
from typing import Optional
from config import config
from Agents.code_generator import CodeGenerator
from Agents.Reflector import Reflector
from Agents.expert_agent import ExpertAgent
from Analysis_pipeline.analysis_agent import AnalysisAgent
from Analysis_pipeline.CSV_statistical_feature_extraction import generate_report
from Analysis_pipeline.data_validation import DataValidator
import Data.plot_script
from Data.plot_script import creat_plot, create_single_dimension_plots

def initialize_components():
    """
    Initialize project components
    """
    # Get configuration
    api_config = config.get_api_config()
    model_config = config.get_model_config()
    path_config = config.get_path_config()

    # Initialize ExpertAgent (for expanding dataset descriptions)
    expert_agent = ExpertAgent(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
        model=model_config.get('expert_model', model_config['code_generator_model'])
    )

    # Initialize CodeGenerator
    code_generator = CodeGenerator(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
        model=model_config['code_generator_model'],
        ts_generator_path=path_config['time_blender_path']
    )

    # Initialize Reflector
    reflector = Reflector(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
        model=model_config['reflector_model']
    )

    # Initialize AnalysisAgent
    analysis_expert = AnalysisAgent(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
        model=model_config['analysis_model'],
    )

    return expert_agent, code_generator, reflector, analysis_expert

def write_log(message: str, log_file_path: Optional[str] = None):
    """
    Logging function
    :param message: Log message
    :param log_file_path: Log file path
    """
    if log_file_path is None:
        log_file_path = config.log_file_path
        
    try:
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception as e:
        print(f"Logging error: {e}")

def validate_and_compare_data(real_data_path: str, synthetic_data_path: str) -> dict:
    """
    Validate and compare real data with synthetic data
    :param real_data_path: Path to real data
    :param synthetic_data_path: Path to synthetic data
    :return: Validation result
    """
    try:
        validation_report = DataValidator.generate_validation_report(
            real_data_path=real_data_path,
            synthetic_data_path=synthetic_data_path
        )
        return validation_report
    except Exception as e:
        print(f"Data validation error: {e}")
        return {"error": str(e)}

def run_generation_loop(
    code_generator,
    reflector,
    analysis_expert,
    data_description: str,
    base_objective: str,
    dataset_path: str,
    output_data_path: str,
    data_caption: str = "",
    max_iterations: int = 10,
    validate_similarity: bool = False,  # Parameter to control similarity validation
    expert_agent = None,  # Expert agent for expanding descriptions
    use_expert_expansion: bool = True  # Whether to use expert expansion
):
    """
    Run generation loop
    :param code_generator: Code generator
    :param reflector: Reflector
    :param analysis_expert: Analysis expert
    :param data_description: Data description
    :param base_objective: Base objective
    :param dataset_path: Dataset path
    :param output_data_path: Output data path
    :param data_caption: Data caption
    :param max_iterations: Maximum iterations
    :param validate_similarity: Whether to validate data similarity
    :param expert_agent: Expert agent for expanding dataset descriptions
    :param use_expert_expansion: Whether to use expert expansion
    """
    iteration_count = 0
    multi_modal = True
    current_data = True

    # Step 0: Use ExpertAgent to expand dataset description
    expanded_description = data_description
    if use_expert_expansion and expert_agent:
        print("===== Step 0: Expert Expansion of Dataset Description =====")
        write_log("===== Step 0: Expert Expansion of Dataset Description =====")
        try:
            print("Expanding dataset description with domain knowledge...")
            expanded_description = expert_agent.expand_description(
                basic_description=data_description,
                dataset_name=data_caption
            )
            print("Dataset description expanded successfully.")
            print("\n--- Expanded Description ---")
            print(expanded_description)
            write_log(f"Expanded description: {expanded_description}")
        except Exception as e:
            print(f"Expert expansion failed: {str(e)}")
            write_log(f"Expert expansion failed: {str(e)}")
            expanded_description = data_description
            print("Using original description instead.")

    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"===== Iteration {iteration_count}: Code Generation & Debugging =====")
        write_log(f"===== Iteration {iteration_count}: Code Generation & Debugging =====")

        # Check if output directory exists, create if not
        output_dir = os.path.dirname(output_data_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if current_data:
            try:
                original_static = generate_report(dataset_path)
                # Create multi-dimension plots for original data (use full 30-day merged.csv)
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                merged_csv_path = os.path.join(repo_root, 'aligned_output', 'merged.csv')
                original_data_multi_plot_dir = os.path.join(os.path.dirname(__file__), 'Dataset', 'CoalMill', 'multi_dimension_plots')
                os.makedirs(original_data_multi_plot_dir, exist_ok=True)
                creat_plot(csv_path=merged_csv_path, output_path=original_data_multi_plot_dir)

                # Create single-dimension plots for original data (use full 30-day merged.csv)
                original_data_single_plot_dir = os.path.join(os.path.dirname(__file__), 'Dataset', 'CoalMill', 'single_dimension_plots')
                os.makedirs(original_data_single_plot_dir, exist_ok=True)
                create_single_dimension_plots(csv_path=merged_csv_path, output_path=original_data_single_plot_dir, max_points=45000)

                # Analyze multi-dimension plots
                original_data_multi_plot_report = analysis_expert.analysis_chat(
                    folder_path=original_data_multi_plot_dir,
                    caption=data_caption,
                    dataset_description=data_description
                )

                # Analyze single-dimension plots
                original_data_single_plot_report = analysis_expert.analysis_chat(
                    folder_path=original_data_single_plot_dir,
                    caption=data_caption,
                    dataset_description=data_description
                )
                
                print("Multi-dimension analysis:")
                print(original_data_multi_plot_report)
                print("Single-dimension analysis:")
                print(original_data_single_plot_report)
                
                current_data_report = (
                    f"Here is the report of Partial data in original dataset."
                    f"Statistical report is here{original_static}\n\n."
                    f"Multi-dimension analysis: {original_data_multi_plot_report}\n\n"
                    f"Single-dimension analysis: {original_data_single_plot_report}\n\n"
                    "You can learn the relationship between variables by these reports."
                )
            except Exception as e:
                print(f"Data analysis failed: {str(e)}")
                write_log(f"Data analysis failed: {str(e)}")
                current_data_report = "Data analysis failed."

        # Step 1.1: Initial code generation and execution
        if iteration_count == 1:
            success, stdout, stderr = code_generator.generate_and_execute(
                expanded_description + current_data_report + base_objective,
                data_output_path=output_data_path  # Pass correct output path
            )
        else:
            # Ensure fix_feedback variable exists
            feedback_content = fix_feedback if 'fix_feedback' in locals() else ""
            response_with_feedback = code_generator.continue_conversation(
                "Here is some feedback about your code. "
                "Refactor or optimize your code with this feedback. "
                + feedback_content
            )
            success, stdout, stderr = code_generator.execute_generated_code(response_with_feedback)

        # Step 1.2: If there are execution errors, let code agent fix until code runs properly
        while stderr:
            print("Initial errors encountered:")
            print(stderr)

            fix_request = (
                f"The generated code produced errors:\n{stderr}\n"
                f"Please fix the code so that it runs properly and generates the expected time series data."
                f"Send the complete fixed code to me."
            )
            new_response = code_generator.continue_conversation(fix_request)
            success, stdout, stderr = code_generator.execute_generated_code(new_response)
            time.sleep(1)

        # At this point, code has run successfully and generated time series data
        print("Code executed successfully. Data generated.")
        write_log("Code executed successfully. Data generated.")

        # Check if generated data file exists
        if not os.path.exists(output_data_path):
            print(f"Warning: Expected output file not found at {output_data_path}")
            write_log(f"Warning: Expected output file not found at {output_data_path}")
            # Continue to next iteration instead of skipping analysis step
        else:
            print(f"Data successfully generated at {output_data_path}")

        print("===== Step 2: Data Analysis =====")
        write_log("===== Step 2: Data Analysis =====")

        # Analyze generated data
        try:
            statistical_report = generate_report(output_data_path)
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            write_log(f"Analysis failed: {str(e)}")
            statistical_report = "Analysis failed. Please check the generated data."

        if multi_modal:
            try:
                # Create multi-dimension plots for generated data for Analysis Agent analysis
                generated_data_multi_plot_dir = os.path.join(os.path.dirname(output_data_path), "multi_dimension_plots")
                creat_plot(csv_path=output_data_path, output_path=generated_data_multi_plot_dir)

                # Create single-dimension plots for generated data for Analysis Agent analysis
                generated_data_single_plot_dir = os.path.join(os.path.dirname(output_data_path), "single_dimension_plots")
                create_single_dimension_plots(csv_path=output_data_path, output_path=generated_data_single_plot_dir, max_points=8000)

                # Analyze multi-dimension plots
                visual_multi_plot_report = analysis_expert.analysis_chat(
                    folder_path=generated_data_multi_plot_dir,
                    dataset_description=data_description
                )

                # Analyze single-dimension plots
                visual_single_plot_report = analysis_expert.analysis_chat(
                    folder_path=generated_data_single_plot_dir,
                    dataset_description=data_description
                )

                print("Generated data multi-dimension analysis:")
                print(visual_multi_plot_report)
                print("Generated data single-dimension analysis:")
                print(visual_single_plot_report)

                write_log(f"Multi-dimension analysis: {visual_multi_plot_report}")
                write_log(f"Single-dimension analysis: {visual_single_plot_report}")

                # Merge two analysis reports
                visual_plot_report = f"Multi-dimension analysis: {visual_multi_plot_report}\n\nSingle-dimension analysis: {visual_single_plot_report}"
            except Exception as e:
                print(f"Visualization failed: {str(e)}")
                write_log(f"Visualization failed: {str(e)}")
                visual_plot_report = "Visualization failed."
                
        print("===== Step 3: Data Validation =====")
        write_log("===== Step 3: Data Validation =====")

        # Validate and compare data (only perform similarity comparison when validate_similarity is True)
        if validate_similarity:
            try:
                validation_report = validate_and_compare_data(dataset_path, output_data_path)
                validation_summary = f"Overall similarity: {validation_report.get('overall_assessment', {}).get('overall_similarity', 'N/A')}"
                print(f"Data validation completed: {validation_summary}")
                write_log(f"Data validation completed: {validation_summary}")
            except Exception as e:
                print(f"Data validation failed: {str(e)}")
                write_log(f"Data validation failed: {str(e)}")
                validation_report = {"error": str(e)}
                validation_summary = "Validation failed"
        else:
            # Simplified validation when not performing similarity comparison
            validation_summary = "Data validation completed: Similarity comparison skipped as requested"
            print(validation_summary)
            write_log(validation_summary)

        print("===== Step 4: Code Evaluation via Reflector =====")
        write_log("===== Step 4: Code Evaluation via Reflector =====")

        # Get code context generated by code agent
        reflector_input = (
            "Please strictly judge if the synthetic data from code_agent meets the requirements and produces realistic time series data as expected."
            "And here is the analysis report on the generated time series data:\n"
            f"Statistical report is here{statistical_report}\n\n."
            f"Data validation report: {validation_summary}\n\n"
        )
        if multi_modal:
            reflector_input += f"After plot the data, the observed information is here{visual_plot_report}\n\n "

        judge_result = reflector.add_message_and_chat(reflector_input)

        print("Reflector Judgment:")
        print(judge_result)
        write_log(f"Reflector Judgment: {str(judge_result)}")

        # Judge Reflector feedback: if "ACCEPTABLE", exit loop; otherwise send feedback to code agent for further improvement
        if "ACCEPTABLE" in judge_result.upper() and "NOT ACCEPTABLE" not in judge_result.upper():
            print("Final code is acceptable. Program ends.")
            write_log("Final code is acceptable. Program ends.")
            break
        else:
            print("Code is not acceptable. Feedback sent to codeagent for further modification.")
            write_log("Code is not acceptable. Feedback sent to codeagent for further modification.")

            fix_feedback = (
                f"The following issues were identified by the Reflector:\n{judge_result}\n"
                "Please modify the code accordingly to address these issues and achieve the expected effects."
            )
            time.sleep(1)