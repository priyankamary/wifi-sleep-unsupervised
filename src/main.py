import argparse
from inference_pipeline import process_user

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_ids", nargs="+", type=int, required=True, help="List of user IDs to process")
    parser.add_argument("--user_data_dir", required=True, help="Path to Fitbit user data directory")
    parser.add_argument("--result_csv", required=True, help="Path to the fitbituserdata.csv file")
    parser.add_argument("--sample_dir", required=True, help="Path to sample dataset directory")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    args = parser.parse_args()

    CONFIG = {
        'user_data_dir': args.user_data_dir,
        'result_csv': args.result_csv,
        'sample_dir': args.sample_dir,
        'save_dir': args.save_dir
    }

    for uid in args.user_ids:
        process_user(uid, CONFIG)