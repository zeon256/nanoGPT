#!/usr/bin/env fish

# Function to print usage/help message
function print_help
    echo "Usage: ./run_benchmark.fish [options]"
    echo ""
    echo "Options:"
    echo "  --number-run N      Number of training runs to perform (N > 0)"
    echo "  --training-script   Path to the training script (default: ../train.py)"
    echo "  --benchmarks        Comma-separated list of benchmarks to run"
    echo "                      Possible values: baseline,mimalloc,mimalloc-thp,jemalloc,jemalloc-thp,tcmalloc,tcmalloc-thp"
    echo "  --num-epoch         Number of epochs for training (N > 0)"
    echo "  --sleep-time        Sleep time in seconds between runs (default: 300)"
    echo "  --help              Display this help message"
    echo ""
    echo "Example:"
    echo "  sudo -E ./run_benchmark.fish --number-run 10 --benchmarks baseline,mimalloc --num-epoch 50 --sleep-time 300"
end

function get_timestamp
    date '+%Y-%m-%d %H:%M:%S'
end

function log_message
    set -l message $argv[1]
    echo "["(get_timestamp)"] $message"
end

function validate_numeric
    set -l value $argv[1]
    set -l param_name $argv[2]

    if not string match -qr '^[0-9]+$' $value
        or test $value -le 0
        echo "Error: $param_name must be a positive integer"
        return 1
    end
    return 0
end

function validate_benchmarks
    set -l benchmarks (string split ',' $argv[1])
    set -l valid_benchmarks baseline mimalloc mimalloc-thp jemalloc jemalloc-thp thp perfg tcmalloc tcmalloc-thp

    for benchmark in $benchmarks
        if not contains $benchmark $valid_benchmarks
            echo "Error: Invalid benchmark '$benchmark'"
            echo "Valid benchmarks are: "(string join ', ' $valid_benchmarks)
            return 1
        end
    end
    return 0
end

# Functions for different benchmark configurations
function setup_baseline
    # No special setup needed for baseline
    return 0
end

# Function to verify THP status
function verify_thp
    set -l expected_status $argv[1]

    # Extract the current status by looking for the value in brackets
    set -l current_status (cat /sys/kernel/mm/transparent_hugepage/enabled | grep -o '\[\w\+\]' | tr -d '[]')

    if test "$current_status" = "$expected_status"
        log_message "THP is correctly set to $expected_status."
    else
        log_message "Error: THP is $current_status, expected $expected_status."
        return 1
    end
end

# Functions for different benchmark configurations
function setup_baseline
    log_message "Setting up baseline configuration."
    return 0
end

function disable_thp
    log_message "Disabling THP."
    echo never >/sys/kernel/mm/transparent_hugepage/enabled

    # Verify and log THP status
    verify_thp never
end

function setup_thp
    log_message "Enabling THP."
    echo always >/sys/kernel/mm/transparent_hugepage/enabled

    # Verify and log THP status
    verify_thp always
end

function setup_mimalloc
    log_message "Setting up mimalloc."
    set -gx LD_PRELOAD /usr/lib/libmimalloc.so
end

function setup_jemalloc
    log_message "Setting up jemalloc."
    set -gx LD_PRELOAD /usr/lib/libjemalloc.so
end

function setup_mimalloc_thp
    setup_mimalloc
    setup_thp
    set -gx MIMALLOC_ALLOW_LARGE_OS_PAGES 1
    log_message "MIMALLOC_ALLOW_LARGE_OS_PAGES set to 1"
end

function setup_jemalloc_thp
    setup_jemalloc
    setup_thp
    set -gx MALLOC_CONF "thp:always,metadata_thp:always,dirty_decay_ms:-1"
end

function setup_perfg
    # do nothing
end

function setup_tcmalloc
    log_message "Setting up tcmalloc."
    set -gx LD_PRELOAD /usr/lib/libtcmalloc.so
end

function cleanup_tcmalloc_thp
    # Reset THP to its default state (assuming 'never' is default)
    echo never >/sys/kernel/mm/transparent_hugepage/enabled
    verify_thp never
end

function cleanup_mimalloc_thp
    # Unset the mimalloc environment variable
    if set -q MIMALLOC_ALLOW_LARGE_OS_PAGES
        set -e MIMALLOC_ALLOW_LARGE_OS_PAGES
        log_message "MIMALLOC_ALLOW_LARGE_OS_PAGES unset"
    end

    # Reset THP to its default state (assuming 'never' is default)
    echo never >/sys/kernel/mm/transparent_hugepage/enabled
    verify_thp never
end

# Function to clean up after using jemalloc with THP
function cleanup_jemalloc_thp
    # Unset the jemalloc configuration
    if set -q MALLOC_CONF
        set -e MALLOC_CONF
        log_message "MALLOC_CONF unset"
    end

    # Reset THP to its default state (assuming 'never' is default)
    echo never >/sys/kernel/mm/transparent_hugepage/enabled
    verify_thp never
end

function cleanup_benchmark
    set -l benchmark $argv[1]
    log_message "Cleaning up after benchmark: $benchmark"

    # Reset LD_PRELOAD if set
    if set -q LD_PRELOAD
        set -e LD_PRELOAD
        log_message "LD_PRELOAD unset"
    end

    # Determine and execute specific cleanups based on the benchmark name
    if string match -q '*mimalloc-thp' $benchmark
        cleanup_mimalloc_thp
    else if string match -q '*jemalloc-thp' $benchmark
        cleanup_jemalloc_thp
    else if string match -q '*tcmalloc-thp' $benchmark
        cleanup_tcmalloc_thp
    else if string match -q '*-thp' $benchmark
        # General THP reset if no specific allocator is used
        echo never >/sys/kernel/mm/transparent_hugepage/enabled
        verify_thp never
    end
end

function run_training_iteration
    set -l run_number $argv[1]
    set -l train_script $argv[2]
    set -l num_epoch $argv[3]
    set -l benchmark $argv[4]
    set -l run_folder $argv[5]

    set date (date +"%Y-%m-%d_%H-%M-%S")
    set output_file "./$run_folder/$benchmark/run_$run_number.txt"
    set json_output_file "./$run_folder/$benchmark/run_$run_number.json"

    log_message "Starting run number $run_number"

    if string match -q perfg $benchmark
        log_message "Setting up for amd_pstate_epp performance and performance scaling"
        setup_perfg
        gamemoderun uv run $train_script config/train_shakespeare_char.py --max_iters=$num_epoch --results_path=$json_output_file &>$output_file
    else
        uv run $train_script config/train_shakespeare_char.py --max_iters=$num_epoch --results_path=$json_output_file &>$output_file
    end

    set time_taken $(jq '.total_time_s' $json_output_file)

    # print out the time taken
    log_message "$benchmark run number $run_number total time taken: $time_taken"

    echo "$benchmark run number $run_number total time taken: $time_taken"

    return 0
end

# Function to run benchmark suite
function run_benchmark
    set -l benchmark $argv[1]
    set -l num_runs $argv[2]
    set -l train_script $argv[3]
    set -l num_epoch $argv[4]
    set -l sleep_time $argv[5]
    set -l run_folder $argv[6]

    set_color --bold blue
    log_message "Starting benchmark: $benchmark"
    set_color normal

    # Create benchmark-specific directory
    set folder_to_make "./$run_folder/$benchmark"
    mkdir $folder_to_make

    # Setup benchmark environment
    switch $benchmark
        case baseline
            disable_thp
            setup_baseline
        case thp
            setup_thp
        case mimalloc
            disable_thp
            setup_mimalloc
        case mimalloc-thp
            setup_mimalloc_thp
        case jemalloc
            setup_jemalloc
        case jemalloc-thp
            setup_jemalloc_thp
        case perfg
            setup_perfg
        case tcmalloc
            disable_thp
            setup_tcmalloc
        case tcmalloc-thp
            setup_tcmalloc_thp
    end

    # Run training iterations
    for i in (seq 1 $num_runs)
        # returns time taken
        set runtime $(run_training_iteration $i $train_script $num_epoch $benchmark $run_folder)

        # Rest between runs (except for the last run)
        if test $i -lt $num_runs
            log_message "Starting $sleep_time second rest period..."
            telegram_send $TELEGRAM_API_KEY $TELEGRAM_CHAT_ID "$runtime[-1]"
            sleep $sleep_time
            log_message "Rest period completed"
        end
    end

    # Cleanup benchmark environment
    cleanup_benchmark $benchmark

    telegram_send $TELEGRAM_API_KEY $TELEGRAM_CHAT_ID "$benchmark completed"

    log_message "Benchmark $benchmark completed"
end

function telegram_send --description 'Send a message to a Telegram group'
    if test (count $argv) -lt 3
        echo "Usage: telegram_send API_KEY CHAT_ID MESSAGE"
        return 1
    end

    set -l api_key $argv[1]
    set -l chat_id $argv[2]
    set -l message $argv[3]

    log_message "Sending to $chat_id"

    set -l url "https://api.telegram.org/bot$api_key/sendMessage"

    set -l response (curl -s -X POST "$url" \
        -d "chat_id=$chat_id" \
        -d "text=$message")

    log_message $response

    if echo $response | jq -e '.ok == true' >/dev/null
        log_message "Message sent successfully"
    else
        log_message "Failed to send message: "(echo $response | jq -r '.description')
    end
end

function main
    # Store original environment variables that might be needed
    set -l original_path $PATH
    set -l original_home $HOME
    set -l original_user $USER

    argparse h/help 'number-run=' 'training-script=' 'benchmarks=' 'num-epoch=' 'sleep-time=' -- $argv
    or return 1

    if set -q _flag_help
        print_help
        return 0
    end

    # Validate required arguments
    if not set -q _flag_number_run
        or not set -q _flag_benchmarks
        or not set -q _flag_num_epoch
        echo "Error: Missing required arguments"
        print_help
        return 1
    end

    # Validate numeric arguments
    validate_numeric $_flag_number_run number-run || return 1
    validate_numeric $_flag_num_epoch num-epoch || return 1

    # Set and validate sleep time
    set sleep_time 300 # Default to 300 seconds (5 minutes)
    if set -q _flag_sleep_time
        validate_numeric $_flag_sleep_time sleep-time || return 1
        set sleep_time $_flag_sleep_time
    end

    # Validate benchmarks
    validate_benchmarks $_flag_benchmarks || return 1

    # Set default training script if not provided
    set train_script "train.py"
    if set -q _flag_training_script
        set train_script $_flag_training_script
    end

    # Check if running as root for THP modifications
    if test (id -u) -ne 0
        log_message "Warning: Script should be run with sudo for THP modifications"
    else
        # If running as root, ensure we have SUDO_USER set
        if not set -q SUDO_USER
            log_message "Error: SUDO_USER environment variable not set. Please run with 'sudo -E' to preserve environment"
            return 1
        end
    end

    # # create run folder based off uname -r
    set run_folder $(uname -r)
    mkdir $run_folder

    # Run each benchmark
    for benchmark in (string split ',' $_flag_benchmarks)
        run_benchmark $benchmark $_flag_number_run $train_script $_flag_num_epoch $sleep_time $run_folder
    end

    log_message "All benchmarks completed"

    # run python script to calculate stats
    set stats_output $(uv run calculate_stats.py "$run_folder" --sample_size 30)

    telegram_send $TELEGRAM_API_KEY $TELEGRAM_CHAT_ID "$stats_output"
    telegram_send $TELEGRAM_API_KEY $TELEGRAM_CHAT_ID "Powering pc off..."

    # poweroff pc lol
    poweroff
end

main $argv
