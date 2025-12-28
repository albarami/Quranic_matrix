"""Run Phase 5 annotation extraction from all 4 tafsirs."""

import os
from src.ai.unified import TafsirAnnotationExtractor

def main():
    extractor = TafsirAnnotationExtractor()
    print("Extracting annotations from all 4 tafsirs...")
    stats = extractor.extract_all()
    
    print("\n=== Extraction Statistics ===")
    print(f"Sources processed: {stats['sources_processed']}")
    print(f"Ayat processed: {stats['ayat_processed']}")
    print(f"Behaviors found: {stats['behaviors_found']}")
    print(f"Relationships found: {stats['relationships_found']}")
    print(f"Inner states found: {stats['inner_states_found']}")
    print(f"Speech acts found: {stats['speech_acts_found']}")
    
    print("\n=== By Source ===")
    for source, source_stats in stats['by_source'].items():
        print(f"  {source}: {source_stats}")
    
    # Save annotations
    os.makedirs("data/annotations", exist_ok=True)
    extractor.save_annotations("data/annotations/tafsir_annotations.jsonl")
    print(f"\nSaved {len(extractor.annotations)} annotations to data/annotations/tafsir_annotations.jsonl")
    
    # Get behavior summary
    summary = extractor.get_behavior_summary()
    print(f"\nUnique behaviors found: {summary['total_behaviors']}")
    print(f"Consensus behaviors (3+ sources): {len(summary['consensus'])}")
    
    # Show top consensus behaviors
    print("\n=== Top Consensus Behaviors ===")
    for beh in summary['consensus'][:10]:
        print(f"  {beh['behavior_id']}: {beh['name_ar']} - {len(beh['sources'])} sources")

if __name__ == "__main__":
    main()
