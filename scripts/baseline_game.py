"""
A simple, text-based Blackjack game implemented with Python best practices.

This script is structured to be both runnable as a standalone game and
importable as a module without side effects. It adheres to PEP 8 styling
and separates core logic from the main game loop.
"""

import os
import random
from typing import List, Tuple

# --- Constants ---
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'Jack': 10, 'Queen': 10, 'King': 10, 'Ace': 11
}

# --- Core Game Logic ---

def get_card_name(card_index: int) -> str:
    """Converts a numerical card index (0-51) to its string name."""
    if not 0 <= card_index <= 51:
        raise ValueError("Card index must be between 0 and 51.")
    suit = SUITS[card_index // 13]
    rank = RANKS[card_index % 13]
    return f"{rank} of {suit}"

def get_card_value(card_index: int) -> int:
    """Gets the Blackjack value for a numerical card index (0-51)."""
    rank = RANKS[card_index % 13]
    return CARD_VALUES[rank]

def calculate_hand_value(hand: List[int]) -> int:
    """
    Calculates the total value of a hand, handling Aces correctly.
    Aces are counted as 11 unless the total exceeds 21, in which case
    they are counted as 1.
    """
    value = sum(get_card_value(card) for card in hand)
    num_aces = sum(1 for card in hand if RANKS[card % 13] == 'Ace')

    while value > 21 and num_aces > 0:
        value -= 10
        num_aces -= 1
    return value

def deal_card() -> int:
    """Selects a single random card from a full 52-card deck."""
    return random.randint(0, 51)

# --- User Interface and Game Flow ---

def clear_screen():
    """Clears the console screen for a cleaner interface."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_hands(player_hand: List[int], dealer_hand: List[int], game_over: bool = False):
    """Displays the cards and scores for both the player and the dealer."""
    clear_screen()
    print("--- Blackjack ---")

    player_score = calculate_hand_value(player_hand)
    print(f"\nYour Hand ({player_score}):")
    for card in player_hand:
        print(f"  {get_card_name(card)}")

    print(f"\nDealer's Hand:")
    if not game_over:
        print(f"  {get_card_name(dealer_hand[0])}")
        print("  [Hidden Card]")
    else:
        dealer_score = calculate_hand_value(dealer_hand)
        for card in dealer_hand:
            print(f"  {get_card_name(card)}")
        print(f"Dealer's Score: {dealer_score}")
    print("-----------------\n")

def get_player_action() -> str:
    """Prompts the player to either 'Hit' or 'Stand' and returns their choice."""
    while True:
        action = input("Type 'h' to Hit or 's' to Stand: ").lower().strip()
        if action in ['h', 's']:
            return action
        print("Invalid input. Please try again.")

def determine_winner(player_score: int, dealer_score: int):
    """Compares final scores and prints the outcome of the game."""
    print("\n--- Game Over ---")
    print(f"Your final score: {player_score}")
    print(f"Dealer's final score: {dealer_score}")

    if player_score > 21:
        print("Bust! You lose.")
    elif dealer_score > 21:
        print("Dealer busts! You win!")
    elif player_score > dealer_score:
        print("You win!")
    elif player_score < dealer_score:
        print("You lose.")
    else:
        print("It's a push (tie).")

def play_round():
    """Manages the logic for a single round of Blackjack."""
    player_hand = [deal_card(), deal_card()]
    dealer_hand = [deal_card(), deal_card()]

    # Player's turn
    while True:
        display_hands(player_hand, dealer_hand)
        player_score = calculate_hand_value(player_hand)
        if player_score >= 21:
            break  # End turn on Blackjack or bust

        action = get_player_action()
        if action == 'h':
            player_hand.append(deal_card())
        elif action == 's':
            break

    player_score = calculate_hand_value(player_hand)

    # Dealer's turn (only if player hasn't busted)
    if player_score <= 21:
        dealer_score = calculate_hand_value(dealer_hand)
        while dealer_score < 17:
            dealer_hand.append(deal_card())
            dealer_score = calculate_hand_value(dealer_hand)

    display_hands(player_hand, dealer_hand, game_over=True)
    determine_winner(player_score, calculate_hand_value(dealer_hand))

def main():
    """The main entry point for the Blackjack game application."""
    while True:
        play_round()
        play_again = input("\nPlay another round? (y/n): ").lower().strip()
        if play_again != 'y':
            print("Thanks for playing!")
            break

# --- Main Execution Guard ---
if __name__ == "__main__":
    main()
