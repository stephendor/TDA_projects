import json
from datetime import datetime

def save_validation_results(results, validation_type, dataset_info=None, output_file_name=None):
    """Save validation results to a JSON file."""
    timestamp = datetime.now().isoformat()
    
    validation_summary = {
        'validation_date': timestamp,
        'validation_type': validation_type,
        'dataset_info': dataset_info if dataset_info else {},
        'results': results,
        'best_performer': None
    }
    
    successful_results = [r for r in results if r.get('status') == 'SUCCESS']
    if successful_results:
        successful_results.sort(key=lambda x: x.get('f1_score', 0), reverse=True)
        validation_summary['best_performer'] = successful_results[0]
        
    if output_file_name is None:
        output_file_name = f"{validation_type.lower().replace(' ', '_')}_results.json"

    with open(output_file_name, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file_name}")
    return output_file_name
