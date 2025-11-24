import re
import json
import os
from collections import defaultdict
from datetime import datetime
import numpy as np

def extract_last_error(log_content):
    """
    Extract the last error from a core.log file content.
    
    Returns:
        tuple: (error_type, full_error_description) or (None, None) if no error found
    """
    lines = log_content.strip().split('\n')
    
    # Look for ERROR lines in reverse order to find the last error
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if '[ERROR]' in line:
            # Extract the error message after the timestamp and logger info
            # Pattern: timestamp [ERROR] logger - error_message
            match = re.search(r'\[ERROR\].*? - (.+)', line)
            if match:
                error_msg = match.group(1)
                
                # Special handling for "Web surfing error" - get the last line of traceback and use it as the error type
                if error_msg == "Web surfing error":
                    # Look for the last line that contains an exception
                    for j in range(len(lines) - 1, i, -1):
                        if any(exc in lines[j] for exc in ['Error:', 'Exception:', 'ValueError:', 'TypeError:', 'JSONDecodeError:', 'SyntaxError:', 'TargetClosedError:']):
                            # Clean up the line to extract just the exception
                            last_exception = lines[j].strip()
                            # Remove leading spaces and common prefixes
                            last_exception = re.sub(r'^\s*', '', last_exception)
                            # Use the specific exception as the error type for better categorization
                            return f"Web surfing error: {last_exception}", f"Web surfing error: {last_exception}"
                    return "Web surfing error", "Web surfing error"
                
                # Handle "Error running task" messages - clean up the prefix
                if "Error running task" in error_msg:
                    # Remove the [Execution xxx] prefix if present
                    clean_msg = re.sub(r'^\[Execution [^\]]+\] ', '', error_msg)
                    return clean_msg, clean_msg
                
                # For other errors like "Error parsing thoughts and action", extract just the main error type
                if error_msg.startswith("Error parsing thoughts and action"):
                    return "Error parsing thoughts and action", "Error parsing thoughts and action"
                elif error_msg.startswith("Invalid action text"):
                    return "Invalid action text", "Invalid action text"
                else:
                    # Extract the first part before any colon for other error types
                    main_error = error_msg.split(':')[0] if ':' in error_msg else error_msg
                    return main_error, main_error
    
    return None, None

def extract_action_timing_stats(log_content):
    """
    Extract timing statistics for WebSurfer actions from log content.
    
    Returns:
        tuple: (avg_time_between_actions_seconds, session_duration_seconds, num_actions)
    """
    lines = log_content.strip().split('\n')
    action_timestamps = []
    session_start = None
    session_end = None
    
    for line in lines:
        # Look for session start
        if '[Execution' in line and '] Start' in line:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                session_start = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
        
        # Look for session end
        if '[Execution' in line and '] Completed' in line:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                session_end = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
        
        # Look for WebSurfer actions (source='WebSurfer')
        if "WebSurferEvent(source='WebSurfer'" in line:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                action_timestamps.append(timestamp)
    
    # Calculate statistics
    avg_time_between_actions = None
    session_duration = None
    num_actions = len(action_timestamps)
    
    if len(action_timestamps) > 1:
        # Calculate average time between consecutive actions
        deltas = []
        for i in range(1, len(action_timestamps)):
            delta = (action_timestamps[i] - action_timestamps[i-1]).total_seconds()
            deltas.append(delta)
        avg_time_between_actions = sum(deltas) / len(deltas)
    
    if session_start and session_end:
        session_duration = (session_end - session_start).total_seconds()
    
    return avg_time_between_actions, session_duration, num_actions

def extract_score_from_json(json_content, heldout_verifiers=False):
    """
    Extract score from JSON content of score files.

    Returns:
        dict or int or None: For '0.8-5-3.json' files, returns a dict with score and success metrics.
                             For other score files, returns just the score (0 or 1).
                             Returns None if parsing fails.
    """
    try:
        data = json.loads(json_content)
        score = data.get('score')

        # Check if this is a 0.8-5-3.json file (has gpt_response_text)
        if 'gpt_response_text' in data:
            # Parse the gpt_response_text to extract success metrics
            gpt_response_text = data.get('gpt_response_text', '{}')
            try:
                if heldout_verifiers:
                    gpt_data = json.loads(gpt_response_text)
                    mm_is_success = gpt_data.get('mm_is_success')
                    rubric_is_success = gpt_data.get('rubric_is_success')
                    verifier_is_success = gpt_data.get('verifier_is_success')

                    # Convert to boolean (handle both int and bool values)
                    mm_is_success_bool = mm_is_success in [1, True, '1'] if mm_is_success is not None else None
                    rubric_is_success_bool = rubric_is_success in [1, True, '1'] if rubric_is_success is not None else None
                    verifier_is_success_bool = verifier_is_success in [1, True, '1'] if verifier_is_success is not None else None

                    return {
                        'score': score if score in [0, 1] else None,
                        'mm_is_success': mm_is_success_bool,
                        'rubric_is_success': rubric_is_success_bool,
                        'verifier_is_success': verifier_is_success_bool,
                        'is_detailed': True
                    }
                else:
                    return {
                        'score': score if score in [0, 1] else None,
                        'is_detailed': False
                    }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing gpt_response_text: {e}")
                # Fall through to simple score return

        # Simple score return for non-detailed score files
        if score in [0, 1]:
            return {'score': score, 'is_detailed': False}
        else:
            print(f"Warning: Invalid score value {score}, expected 0 or 1")
            return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON: {e}")
        return None

def extract_final_answer_data(json_content):
    """
    Extract final answer data from _final_answer.json content.

    Returns:
        dict: Contains is_aborted, final_answer, num_screenshots, and token_usage
    """
    try:
        data = json.loads(json_content)
        is_aborted = data.get('is_aborted', False)
        final_answer = data.get('final_answer', '')
        screenshots = data.get('screenshots', [])
        num_screenshots = len(screenshots) if screenshots else 0

        # Extract token usage if available
        token_usage = data.get('token_usage', None)

        return {
            'is_aborted': is_aborted,
            'final_answer': final_answer,
            'num_screenshots': num_screenshots,
            'token_usage': token_usage
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing final_answer JSON: {e}")
        return None

def aggregate_post_eval_errors(folders, long_session_threshold_seconds=1800, num_actions=30):
    # Re-initialize error tracking dictionaries with the updated function
    error_counts = defaultdict(int)
    error_trajectories = defaultdict(list)
    no_error_count = 0
    total_trajectories = len(folders)
    
    # Timing statistics tracking
    all_action_deltas = []
    all_session_durations = []
    all_num_actions_all = []
    all_num_actions_non_aborted = []
    total_sessions_with_timing = 0
    
    # Score tracking
    all_scores = []
    avg_score = 0.0
    trajectories_with_scores = 0

    # Detailed success metrics tracking (for 0.8-5-3.json files)
    mm_is_success_count = 0
    rubric_is_success_count = 0
    verifier_is_success_count = 0
    mm_and_verifier_agree_count = 0
    detailed_scores_count = 0
    
    # Final answer tracking
    is_aborted_true_count = 0
    is_aborted_false_count = 0
    empty_answer_below_action_budget_but_not_aborted = 0
    trajectories_with_final_answer = 0
    final_answer_file_not_found = 0

    # Token usage tracking
    all_prompt_tokens = []
    all_completion_tokens = []
    trajectories_with_token_usage = 0

    print(f"Re-processing {total_trajectories} trajectories with updated error extraction...")

    # Process all trajectory folders
    for i, item in enumerate(folders):
        if i % 100 == 0:
            print(f"Processed {i}/{total_trajectories} trajectories...")
        
        # Get base name of the trajectory
        item_id = item['name'].name
        
        # Initialize trajectory-specific variables
        trajectory_num_actions = None
        trajectory_is_aborted = None
        
        # Find core.log file
        core_log_file = next((f for f in item['files'] if 'core.log' in str(f)), None)
        
        # Look for score files in scores subdirectory
        score_file = None
        scores_dir = item['name'] / 'scores'
        if scores_dir.exists():
            for score_filename in ['0.8-5-3.json', 'gpt_eval.json']:
                potential_score_file = scores_dir / score_filename
                if potential_score_file.exists():
                    score_file = potential_score_file
                    break

        uses_heldout_verifiers = True if score_file and '0.8-5-3.json' in str(score_file) else False
        if not uses_heldout_verifiers:
            assert score_file is None or 'gpt_eval.json' in str(score_file), f"Unexpected score file found: {score_file}"
        
        if core_log_file:
            try:
                with open(core_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    
                # Extract the last error
                error_type, _ = extract_last_error(log_content)
                
                # Extract timing statistics
                avg_delta, session_duration, num_actions = extract_action_timing_stats(log_content)
                trajectory_num_actions = num_actions
                
                # Collect timing data for overall statistics
                if avg_delta is not None:
                    all_action_deltas.append(avg_delta)
                if session_duration is not None:
                    all_session_durations.append(session_duration)
                    total_sessions_with_timing += 1
                if num_actions is not None:
                    all_num_actions_all.append(num_actions)
                    
                    # Check for long sessions
                    if session_duration is not None and session_duration > long_session_threshold_seconds:
                        # Add Warning-Long Session as an error type
                        warning_type = f"Long Session Warning - Session Exceeded {long_session_threshold_seconds} seconds"
                        error_counts[warning_type] += 1
                        error_trajectories[warning_type].append(item_id)
                
                if error_type:
                    error_counts[error_type] += 1
                    error_trajectories[error_type].append(item_id)
                else:
                    no_error_count += 1
                    
            except Exception as e:
                print(f"Error reading {core_log_file}: {e}")
        else:
            print(f"No core.log found for {item_id}")
            
        # Process score file if found
        if score_file:
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_content = f.read()
                    score_data = extract_score_from_json(score_content, heldout_verifiers=uses_heldout_verifiers)
                    if score_data is not None:
                        score = score_data.get('score')
                        if score is not None:
                            all_scores.append(score)
                            trajectories_with_scores += 1

                        # Track detailed success metrics if available
                        if score_data.get('is_detailed', False):
                            detailed_scores_count += 1
                            mm_success = score_data.get('mm_is_success') is True
                            verifier_success = score_data.get('verifier_is_success') is True

                            if mm_success:
                                mm_is_success_count += 1
                            if score_data.get('rubric_is_success') is True:
                                rubric_is_success_count += 1
                            if verifier_success:
                                verifier_is_success_count += 1

                            # Track when both mm_is_success AND verifier_is_success are True
                            if mm_success and verifier_success:
                                mm_and_verifier_agree_count += 1
            except Exception as e:
                print(f"Error reading score file {score_file}: {e}")
                
        # Look for _final_answer.json file (with ID prefix)
        final_answer_file = None
        for file_path in item['files']:
            if str(file_path).endswith('_final_answer.json'):
                final_answer_file = file_path
                break
        
        # Process final_answer file if found
        if final_answer_file:
            try:
                with open(final_answer_file, 'r', encoding='utf-8') as f:
                    final_answer_content = f.read()
                    final_answer_data = extract_final_answer_data(final_answer_content)
                    if final_answer_data is not None:
                        trajectories_with_final_answer += 1
                        trajectory_is_aborted = final_answer_data['is_aborted']

                        # Count is_aborted cases
                        if final_answer_data['is_aborted']:
                            is_aborted_true_count += 1
                        else:
                            is_aborted_false_count += 1

                            # Check for empty answer below action budget condition
                            final_answer = final_answer_data['final_answer']
                            num_screenshots = final_answer_data['num_screenshots']

                            if (final_answer in ['<no_answer>', None, ''] and
                                num_screenshots <= num_actions):
                                empty_answer_below_action_budget_but_not_aborted += 1

                        # Extract token usage if available
                        token_usage = final_answer_data.get('token_usage')
                        if token_usage is not None and isinstance(token_usage, dict):
                            # Sum up tokens from all sources in token_usage
                            total_prompt_tokens = 0
                            total_completion_tokens = 0

                            for source, usage in token_usage.items():
                                if isinstance(usage, dict):
                                    prompt_tokens = usage.get('prompt_tokens', 0)
                                    completion_tokens = usage.get('completion_tokens', 0)

                                    total_prompt_tokens += prompt_tokens
                                    total_completion_tokens += completion_tokens

                            if total_prompt_tokens > 0 or total_completion_tokens > 0:
                                all_prompt_tokens.append(total_prompt_tokens)
                                all_completion_tokens.append(total_completion_tokens)
                                trajectories_with_token_usage += 1

            except Exception as e:
                print(f"Error reading final_answer file {final_answer_file}: {e}")
        else:
            # Count trajectories where final_answer file was not found
            final_answer_file_not_found += 1
            
        # Add to non-aborted actions list if trajectory was not aborted and has action data
        if trajectory_num_actions is not None and trajectory_is_aborted is not None and not trajectory_is_aborted:
            all_num_actions_non_aborted.append(trajectory_num_actions)

    print(f"Completed processing {total_trajectories} trajectories!")
    print(f"Trajectories with no errors: {no_error_count}")
    print(f"Trajectories with errors: {sum(error_counts.values())}")
    
    # Print score statistics
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"Average score: {avg_score:.3f} (from {len(all_scores)} trajectories with scores)")
    else:
        print("No score files found or processed")

    # Print detailed success metrics (for 0.8-5-3.json files)
    if detailed_scores_count > 0:
        print(f"\nDetailed success metrics (from {detailed_scores_count} trajectories with 0.8-5-3.json scores):")
        mm_success_pct = (mm_is_success_count / detailed_scores_count) * 100
        rubric_success_pct = (rubric_is_success_count / detailed_scores_count) * 100
        verifier_success_pct = (verifier_is_success_count / detailed_scores_count) * 100
        mm_and_verifier_pct = (mm_and_verifier_agree_count / detailed_scores_count) * 100
        print(f"  mm_is_success == True: {mm_is_success_count}/{detailed_scores_count} ({mm_success_pct:.1f}%)")
        print(f"  rubric_is_success == True: {rubric_is_success_count}/{detailed_scores_count} ({rubric_success_pct:.1f}%)")
        print(f"  verifier_is_success == True: {verifier_is_success_count}/{detailed_scores_count} ({verifier_success_pct:.1f}%)")
        print(f"  mm_is_success AND verifier_is_success == True: {mm_and_verifier_agree_count}/{detailed_scores_count} ({mm_and_verifier_pct:.1f}%)")
    else:
        print("\nNo detailed success metrics found (no 0.8-5-3.json files processed)")
        
    # Print final answer statistics
    print(f"Final answer statistics:")
    print(f"  final_answer_file_not_found: {final_answer_file_not_found}")
    if trajectories_with_final_answer > 0:
        print(f"  From {trajectories_with_final_answer} trajectories with final_answer files:")
        print(f"    is_aborted == True: {is_aborted_true_count}")
        print(f"    is_aborted == False: {is_aborted_false_count}")
        print(f"    empty_answer_below_action_budget_but_not_aborted: {empty_answer_below_action_budget_but_not_aborted}")
    else:
        print("  No final_answer files found or processed")

    # Print token usage statistics
    print(f"\nToken usage statistics:")
    if trajectories_with_token_usage > 0:
        sum_prompt_tokens = sum(all_prompt_tokens)
        sum_completion_tokens = sum(all_completion_tokens)
        avg_prompt_tokens = sum_prompt_tokens / len(all_prompt_tokens)
        avg_completion_tokens = sum_completion_tokens / len(all_completion_tokens)
        std_prompt_tokens = np.std(all_prompt_tokens)
        std_completion_tokens = np.std(all_completion_tokens)

        print(f"  Trajectories with token usage: {trajectories_with_token_usage}/{total_trajectories}")
        print(f"  Total prompt tokens: {sum_prompt_tokens:,}")
        print(f"  Total completion tokens: {sum_completion_tokens:,}")
        print(f"  Total tokens: {sum_prompt_tokens + sum_completion_tokens:,}")
        print(f"  Average prompt tokens per task: {avg_prompt_tokens:.2f} ± {std_prompt_tokens:.2f}")
        print(f"  Average completion tokens per task: {avg_completion_tokens:.2f} ± {std_completion_tokens:.2f}")
        print(f"  Average total tokens per task: {avg_prompt_tokens + avg_completion_tokens:.2f} ± {np.std([p + c for p, c in zip(all_prompt_tokens, all_completion_tokens)]):.2f}")
    else:
        print(f"  No token usage data found in final_answer files")
    
    # Print timing statistics with standard deviation
    if all_action_deltas:
        avg_action_delta = sum(all_action_deltas) / len(all_action_deltas)
        std_action_delta = (sum((x - avg_action_delta) ** 2 for x in all_action_deltas) / len(all_action_deltas)) ** 0.5
        print(f"A) Average time between actions: {avg_action_delta:.2f} ± {std_action_delta:.2f} seconds (across {len(all_action_deltas)} sessions with actions)")
    else:
        print("A) Average time between actions: N/A (no sessions with multiple actions found)")
        
    if all_session_durations:
        avg_session_duration = sum(all_session_durations) / len(all_session_durations)
        std_session_duration = (sum((x - avg_session_duration) ** 2 for x in all_session_durations) / len(all_session_durations)) ** 0.5
        print(f"B) Average session duration: {avg_session_duration:.2f} ± {std_session_duration:.2f} seconds ({avg_session_duration/60:.2f} ± {std_session_duration/60:.2f} minutes) (across {len(all_session_durations)} sessions)")
    else:
        print("B) Average session duration: N/A (no sessions with timing data found)")
        
    if all_num_actions_all:
        avg_num_actions_all = sum(all_num_actions_all) / len(all_num_actions_all)
        std_num_actions_all = (sum((x - avg_num_actions_all) ** 2 for x in all_num_actions_all) / len(all_num_actions_all)) ** 0.5
        print(f"C) Average number of actions per trajectory (all): {avg_num_actions_all:.2f} ± {std_num_actions_all:.2f} actions (across {len(all_num_actions_all)} trajectories)")
    else:
        print("C) Average number of actions per trajectory (all): N/A (no trajectories with action data found)")
        
    if all_num_actions_non_aborted:
        avg_num_actions_non_aborted = sum(all_num_actions_non_aborted) / len(all_num_actions_non_aborted)
        std_num_actions_non_aborted = (sum((x - avg_num_actions_non_aborted) ** 2 for x in all_num_actions_non_aborted) / len(all_num_actions_non_aborted)) ** 0.5
        print(f"D) Average number of actions per trajectory (non-aborted): {avg_num_actions_non_aborted:.2f} ± {std_num_actions_non_aborted:.2f} actions (across {len(all_num_actions_non_aborted)} non-aborted trajectories)")
    else:
        print("D) Average number of actions per trajectory (non-aborted): N/A (no non-aborted trajectories with action data found)")

    # Generate the final error histogram table using pandas
    import pandas as pd

    print("\n" + "="*120)
    print("ERROR ANALYSIS SUMMARY (UPDATED)")
    print("="*120)

    # Create a detailed table with error counts and sample trajectories
    error_data = []
    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        # Get first few trajectory examples (limit to 3 for readability)
        sample_trajectories = error_trajectories[error_type][:3]
        sample_str = ", ".join(sample_trajectories)
        if len(error_trajectories[error_type]) > 3:
            sample_str += f", ... (+{len(error_trajectories[error_type])-3} more)"
        
        error_data.append({
            'Error Type': error_type,
            'Count': count,
            'Percentage': f"{(count/total_trajectories)*100:.1f}%",
            'Sample Trajectories': sample_str
        })

    if all_scores:
        error_data.append({
            'Error Type': f"Average Score for {len(all_scores)}/{total_trajectories}",
            'Count': f"{avg_score:.3f}",
            'Percentage': f"{(len(all_scores)/total_trajectories)*100:.1f}%",
            'Sample Trajectories': ""
        })

    # Add detailed success metrics to the table
    if detailed_scores_count > 0:
        mm_success_ratio = mm_is_success_count / detailed_scores_count
        rubric_success_ratio = rubric_is_success_count / detailed_scores_count
        verifier_success_ratio = verifier_is_success_count / detailed_scores_count
        mm_and_verifier_ratio = mm_and_verifier_agree_count / detailed_scores_count
        detailed_scores_pct = (detailed_scores_count / total_trajectories) * 100

        error_data.append({
            'Error Type': f"mm_is_success == True",
            'Count': mm_success_ratio,
            'Percentage': f"{detailed_scores_pct:.1f}%",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"rubric_is_success == True",
            'Count': rubric_success_ratio,
            'Percentage': f"{detailed_scores_pct:.1f}%",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"verifier_is_success == True",
            'Count': verifier_success_ratio,
            'Percentage': f"{detailed_scores_pct:.1f}%",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"mm_is_success AND verifier_is_success == True",
            'Count': mm_and_verifier_ratio,
            'Percentage': f"{detailed_scores_pct:.1f}%",
            'Sample Trajectories': ""
        })
        
    # Add final answer statistics to the table
    error_data.append({
        'Error Type': f"final_answer_file_not_found",
        'Count': final_answer_file_not_found,
        'Percentage': f"{(final_answer_file_not_found/total_trajectories)*100:.1f}%",
        'Sample Trajectories': ""
    })
    
    if trajectories_with_final_answer > 0:
        error_data.append({
            'Error Type': f"is_aborted == True",
            'Count': is_aborted_true_count,
            'Percentage': f"{(is_aborted_true_count/total_trajectories)*100:.1f}%",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"is_aborted == False", 
            'Count': is_aborted_false_count,
            'Percentage': f"{(is_aborted_false_count/total_trajectories)*100:.1f}%",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"empty_answer_below_action_budget_but_not_aborted",
            'Count': empty_answer_below_action_budget_but_not_aborted,
            'Percentage': f"{(empty_answer_below_action_budget_but_not_aborted/total_trajectories)*100:.1f}%",
            'Sample Trajectories': ""
        })

    # Add token usage statistics to the table
    if trajectories_with_token_usage > 0:
        sum_prompt_tokens = sum(all_prompt_tokens)
        sum_completion_tokens = sum(all_completion_tokens)
        avg_prompt_tokens = sum_prompt_tokens / len(all_prompt_tokens)
        avg_completion_tokens = sum_completion_tokens / len(all_completion_tokens)
        std_prompt_tokens = np.std(all_prompt_tokens)
        std_completion_tokens = np.std(all_completion_tokens)
        all_total_tokens = [p + c for p, c in zip(all_prompt_tokens, all_completion_tokens)]
        avg_total_tokens = sum(all_total_tokens) / len(all_total_tokens)
        std_total_tokens = np.std(all_total_tokens)

        error_data.append({
            'Error Type': f"total_prompt_tokens",
            'Count': sum_prompt_tokens,
            'Percentage': f"{trajectories_with_token_usage}/{total_trajectories}",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"total_completion_tokens",
            'Count': sum_completion_tokens,
            'Percentage': f"{trajectories_with_token_usage}/{total_trajectories}",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"total_tokens",
            'Count': sum_prompt_tokens + sum_completion_tokens,
            'Percentage': f"{trajectories_with_token_usage}/{total_trajectories}",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"avg_prompt_tokens_per_task: {avg_prompt_tokens:.2f} ± {std_prompt_tokens:.2f}",
            'Count': avg_prompt_tokens,
            'Percentage': f"{trajectories_with_token_usage}/{total_trajectories}",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"avg_completion_tokens_per_task: {avg_completion_tokens:.2f} ± {std_completion_tokens:.2f}",
            'Count': avg_completion_tokens,
            'Percentage': f"{trajectories_with_token_usage}/{total_trajectories}",
            'Sample Trajectories': ""
        })
        error_data.append({
            'Error Type': f"avg_total_tokens_per_task: {avg_total_tokens:.2f} ± {std_total_tokens:.2f}",
            'Count': avg_total_tokens,
            'Percentage': f"{trajectories_with_token_usage}/{total_trajectories}",
            'Sample Trajectories': ""
        })

    # Add action statistics to the table
    if all_num_actions_all:
        avg_num_actions_all = sum(all_num_actions_all) / len(all_num_actions_all)
        std_num_actions_all = (sum((x - avg_num_actions_all) ** 2 for x in all_num_actions_all) / len(all_num_actions_all)) ** 0.5
        error_data.append({
            'Error Type': f"average_num_actions_all: {avg_num_actions_all:.2f} ± {std_num_actions_all:.2f}",
            'Count': avg_num_actions_all,
            'Percentage': f"{len(all_num_actions_all)}/{total_trajectories}",
            'Sample Trajectories': ""
        })

    if all_num_actions_non_aborted:
        avg_num_actions_non_aborted = sum(all_num_actions_non_aborted) / len(all_num_actions_non_aborted)
        std_num_actions_non_aborted = (sum((x - avg_num_actions_non_aborted) ** 2 for x in all_num_actions_non_aborted) / len(all_num_actions_non_aborted)) ** 0.5
        error_data.append({
            'Error Type': f"average_num_actions_non_aborted: {avg_num_actions_non_aborted:.2f} ± {std_num_actions_non_aborted:.2f}",
            'Count': avg_num_actions_non_aborted,
            'Percentage': f"{len(all_num_actions_non_aborted)}/{total_trajectories}",
            'Sample Trajectories': ""
        })

    # Create pandas DataFrame
    df = pd.DataFrame(error_data)

    # Display the DataFrame
    print(f"\nTotal trajectories analyzed: {total_trajectories}")
    print(f"Trajectories with errors: {sum(error_counts.values())} ({(sum(error_counts.values())/total_trajectories)*100:.1f}%)")
    print(f"Trajectories without errors: {no_error_count} ({(no_error_count/total_trajectories)*100:.1f}%)")
    print("\nError Breakdown:")

    # Display the DataFrame
    return df

def count_web_surfer_log_entries(folders):
    """
    Count the number of WebSurferEvent log entries in web_surfer.log files
    and return detailed statistics about the number of steps per trajectory,
    split by aborted vs non-aborted trajectories.

    Args:
        folders: List of folder dictionaries with 'name' and 'files' keys

    Returns:
        dict: Dictionary containing step statistics split by aborted flag, or None if no log files found
    """

    step_counts_aborted = []
    step_counts_not_aborted = []
    trajectories_with_logs = 0

    for item in folders:
        # Look for web_surfer.log file
        web_surfer_log_file = None
        for file_path in item['files']:
            if str(file_path).endswith('web_surfer.log'):
                web_surfer_log_file = file_path
                break

        # Look for _final_answer.json file to get aborted status
        final_answer_file = None
        for file_path in item['files']:
            if str(file_path).endswith('_final_answer.json'):
                final_answer_file = file_path
                break

        if web_surfer_log_file and web_surfer_log_file.exists():
            try:
                with open(web_surfer_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()

                # Count actions the same way as in trajectory.py
                # Count events with source='WebSurfer' and action field (not None)
                # This matches the logic in trajectory.py line 70 and 79
                step_count = 0
                for line in log_content.strip().split('\n'):
                    if line.strip():
                        try:
                            event = json.loads(line)
                            if event.get('source', None) == "WebSurfer" and event.get('action', None) is not None:
                                step_count += 1
                        except json.JSONDecodeError:
                            # Skip lines that aren't valid JSON
                            pass

                # Determine if trajectory was aborted
                # Default to True if we can't determine (safer assumption)
                is_aborted = True
                if final_answer_file and final_answer_file.exists():
                    try:
                        with open(final_answer_file, 'r', encoding='utf-8') as f:
                            final_answer_content = f.read()
                            final_answer_data = extract_final_answer_data(final_answer_content)
                            if final_answer_data is not None:
                                is_aborted = final_answer_data['is_aborted']
                            # If data parsing failed, assume aborted (default)
                    except Exception as e:
                        print(f"Error reading final_answer file {final_answer_file}: {e} - assuming aborted")
                # If no final_answer file exists, assume aborted (default)

                # Add to appropriate list
                if is_aborted:
                    step_counts_aborted.append(step_count)
                else:
                    step_counts_not_aborted.append(step_count)

                trajectories_with_logs += 1

            except Exception as e:
                print(f"Error reading {web_surfer_log_file}: {e}")

    def calculate_stats_for_group(step_counts_list, group_name):
        """Helper function to calculate statistics for a group of trajectories."""
        if len(step_counts_list) == 0:
            return None

        step_counts = np.array(step_counts_list)

        # Filter to only include trajectories with at least 1 step
        step_counts = step_counts[step_counts >= 1]
        num_trajectories = len(step_counts)

        if num_trajectories == 0:
            return None

        # Calculate statistics
        avg_steps = float(np.mean(step_counts))
        min_steps = int(np.min(step_counts))
        max_steps = int(np.max(step_counts))
        median_steps = float(np.median(step_counts))

        # Count trajectories with exactly 1 step
        count_one_step = int(np.sum(step_counts == 1))

        # Count trajectories at max steps
        count_max_steps = int(np.sum(step_counts == max_steps))

        # Create histogram with bins spanning the actual data (1, 2, 3, ...)
        bins = list(range(min_steps, max_steps + 2))  # min_steps to max_steps+1, inclusive
        bin_labels = [str(i) for i in range(min_steps, max_steps + 1)]
        hist_counts = []

        for i in range(len(bins) - 1):
            count = int(np.sum(step_counts == bins[i]))
            hist_counts.append(count)

        # Create histogram as list of [label, count] pairs
        histogram = [[label, count] for label, count in zip(bin_labels, hist_counts)]

        return {
            'avg_steps': avg_steps,
            'min_steps': min_steps,
            'max_steps': max_steps,
            'median_steps': median_steps,
            'count_one_step': count_one_step,
            'count_max_steps': count_max_steps,
            'histogram': histogram,
            'total_trajectories': num_trajectories,
            'total_steps': int(np.sum(step_counts))
        }

    if trajectories_with_logs > 0:
        # Calculate statistics for both groups
        stats_aborted = calculate_stats_for_group(step_counts_aborted, "aborted")
        stats_not_aborted = calculate_stats_for_group(step_counts_not_aborted, "not_aborted")

        # Combine overall statistics (for backwards compatibility)
        all_step_counts = step_counts_aborted + step_counts_not_aborted
        stats_overall = calculate_stats_for_group(all_step_counts, "overall")

        print(f"Found {trajectories_with_logs} trajectories with web_surfer.log files")
        print(f"  Aborted: {len(step_counts_aborted)}")
        print(f"  Not aborted: {len(step_counts_not_aborted)}")

        return {
            'overall': stats_overall,
            'aborted': stats_aborted,
            'not_aborted': stats_not_aborted
        }
    else:
        return None