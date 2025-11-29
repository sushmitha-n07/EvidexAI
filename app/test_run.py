from agent.scene_agent import SceneAgent

def main():
    # Sample input
    fir_text = "Victim found in kitchen, blunt force trauma, no weapon recovered. Possible domestic situation."
    detected_objects = ["bat", "chair", "refrigerator"]

    # Initialize agent and analyze
    agent = SceneAgent()
    info, similar_cases, reasoning = agent.analyze(fir_text, detected_objects)

    # Display results
    print("Extracted Info:", info)
    print("\nSimilar Cases:")
    for case in similar_cases:
        print("-", case)
    print("\nReasoning:\n", reasoning)

if __name__ == "__main__":
    main()