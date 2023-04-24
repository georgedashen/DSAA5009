nohup python baseline.py --top_n 50 --k_user 1000 --output ./result_negative_test --negative ./test_negative_500 --test &
nohup python baseline.py --top_n 50 --model random --output ./result_negative_test --negative ./test_negative_500 --test &
nohup python baseline.py --top_n 50 --model majority --output ./result_negative_test --negative ./test_negative_500 --test &

