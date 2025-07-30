# /graph_analysis/run_graph_integration.py

from data_ingestion.platform_connectors import load_sample_chats
from backend import chat_analysis
from graph_analysis import temporal_graph_builder, risk_scorer
from backend import gnn_model, intervention, red_team_simulator

def main():
    # 1. Ingest sample chats
    chats = load_sample_chats(num_chats=3)

    # 2. Analyze each message
    analyzed_messages = []
    for chat_idx, chat in enumerate(chats, 1):
        print(f"\n=== Chat Session {chat_idx} [{chat.get('platform', 'unknown')}] ===")
        for msg_idx, msg in enumerate(chat.get('messages', []), 1):
            analysis = chat_analysis.analyze_chat_message(msg['text'])
            analysis['sender'] = msg['sender']
            analysis['receiver'] = msg['receiver']
            analysis['timestamp'] = msg.get('timestamp')
            analyzed_messages.append(analysis)

    if not analyzed_messages:
        print("No messages found for analysis. Exiting.")
        return

    # 3. Build the temporal social graph
    print("\nBuilding temporal social graph...")
    social_graph = temporal_graph_builder.build_temporal_graph(analyzed_messages)
    print(f"Graph summary: {getattr(social_graph, 'num_nodes', 0)} nodes, {getattr(social_graph, 'num_edges', 0)} edges")

    # 4. Run GNN inference
    print("\nRunning GNN inference for network-level risk detection...")
    gnn_results = gnn_model.run_gnn_inference(social_graph)
    preds = gnn_results.get('predictions', [])
    probs = gnn_results.get('probabilities', [])
    node_map = getattr(social_graph, 'node_map', {})
    index_to_user = {idx: user for user, idx in node_map.items()}

    print("\n" + "="*60)
    print("Network-level Risk Output (GNN Results):")
    print(gnn_results)
    print("="*60 + "\n")

    # 5. Red Team Simulation & Composite Risk Scoring
    print("Initiating Red Team Simulation for flagged users...")
    agent = red_team_simulator.RedTeamAgent()
    flagged = False

    for idx, (pred, prob) in enumerate(zip(preds, probs)):
        user_id = index_to_user.get(idx)
        if user_id and pred == 1:  # Only analyze users flagged as suspicious
            flagged = True
            print(f"\nRed-teaming user {user_id} (network risk: {prob:.2f})")
            demo_prompts = [
                "Hey, are you available for a private chat?",
                "I have something sensitive to share.",
            ]
            red_team_results = agent.simulate_multi_turn_conversation(demo_prompts)
            for turn in red_team_results:
                print(f"RT Turn {turn['turn']} Prompt: {turn['prompt']}")
                print(f"Response: {turn['response']}")
                print(f"Simulated risk score: {turn['risk_score']:.3f}")

            max_sim_risk = max([turn['risk_score'] for turn in red_team_results]) if red_team_results else 0

            heuristic_flags = {}
            for turn in red_team_results:
                if turn['risk_score'] > 0.5:
                    heuristic_flags['red_team_suspicion'] = heuristic_flags.get('red_team_suspicion', 0) + 1

            composite = risk_scorer.compute_composite_risk(
                gnn_prob=prob,
                heuristic_flags=heuristic_flags,
                red_team_risk=max_sim_risk
            )
            print(f"Composite risk: {composite['composite_score']:.2f}")
            print("Composite explanation:", composite['explanation'])

            result = intervention.intervention_policy(
                user_id=user_id,
                message="RedTeam Simulation + GNN + Heuristics: Composite high social engineering risk.",
                risk_score=composite['composite_score'],
                context={
                    "network_probability": float(prob),
                    "red_team_max_sim_risk": float(max_sim_risk),
                    "composite_explanation": composite['explanation']
                }
            )
            if result:
                print(f"INTERVENTION for user {user_id}:")
                print(f"  Action: {result['action']}")
                print(f"  Message: {result['message']}")
                print(f"  Risk: {result['risk_score']:.2f}")
                print(f"  Time: {result['timestamp']}")
                print("-" * 40)
            else:
                print(f"No intervention needed for {user_id} after composite risk scoring.")

    if not flagged:
        print("No flagged users for red team simulation or intervention in this run.")

if __name__ == "__main__":
    main()
