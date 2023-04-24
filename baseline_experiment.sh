# user-based kNN model
nohup python baseline.py --k_user 5 --top_n 50 --test --output ./result_test &
nohup python baseline.py --k_user 10 --top_n 50 --test --output ./result_test &
nohup python baseline.py --k_user 20 --top_n 50 --test --output ./result_test &
nohup python baseline.py --k_user 10 --top_n 10 --test --output ./result_test &
nohup python baseline.py --k_user 10 --top_n 20 --test --output ./result_test &
## following two are used in the report
nohup python baseline.py --k_user 100 --top_n 50 --test --output ./result_test &
nohup python baseline.py --k_user 1000 --top_n 50 --test --output ./result_test &

# random model
nohup python baseline.py --model random --top_n 10 --test --output ./result_test &
nohup python baseline.py --model random --top_n 20 --test --output ./result_test &
## following is used in the report
nohup python baseline.py --model random --top_n 50 --test --output ./result_test &

# random model
nohup python baseline.py --model random --top_n 10 --test --output ./result_test &
nohup python baseline.py --model random --top_n 20 --test --output ./result_test &
## following is used in the report
nohup python baseline.py --model random --top_n 50 --test --output ./result_test &

# majority model
## following is used in the report
nohup python baseline.py --model majority --top_n 50 --test --output ./result_test &
