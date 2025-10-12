import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueBettingStrategy:
    """
    Conservative betting strategy for non-major soccer leagues
    
    Key Features:
    - Conservative Kelly Criterion implementation
    - Dynamic position sizing
    - Risk management controls
    - Market impact consideration
    - Confidence-based betting
    - Drawdown protection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize betting strategy
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.betting_history = []
        self.current_capital = 0
        self.peak_capital = 0
        self.current_drawdown = 0
        self.betting_stats = {}
        
    def setup_logging(self):
        """Setup logging for betting strategy"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load betting strategy configuration"""
        if config is None:
            self.config = {
                'capital_management': {
                    'initial_capital': 1000,
                    'min_capital': 100,
                    'max_position_size': 0.1,  # 10% of capital
                    'min_position_size': 0.01,  # 1% of capital
                    'reserve_capital': 0.2      # 20% reserve
                },
                'kelly_criterion': {
                    'enabled': True,
                    'kelly_fraction': 0.1,     # Conservative 10% of optimal Kelly
                    'max_kelly': 0.05,         # Maximum 5% Kelly
                    'min_kelly': 0.005,        # Minimum 0.5% Kelly
                    'kelly_decay': 0.95,       # Kelly decay factor
                    'kelly_recovery': 1.05     # Kelly recovery factor
                },
                'confidence_filtering': {
                    'min_confidence': 0.6,
                    'max_confidence': 0.95,
                    'confidence_threshold': 0.7,
                    'confidence_decay': 0.98,
                    'confidence_recovery': 1.02
                },
                'odds_filtering': {
                    'min_odds': 1.5,
                    'max_odds': 10.0,
                    'preferred_odds_range': [2.0, 5.0],
                    'odds_weight': 0.3
                },
                'risk_management': {
                    'max_drawdown': 0.2,       # 20% max drawdown
                    'stop_loss': 0.15,         # 15% stop loss
                    'take_profit': 0.3,         # 30% take profit
                    'daily_loss_limit': 0.05,  # 5% daily loss limit
                    'weekly_loss_limit': 0.15, # 15% weekly loss limit
                    'monthly_loss_limit': 0.25 # 25% monthly loss limit
                },
                'betting_limits': {
                    'max_bets_per_day': 5,
                    'max_bets_per_week': 20,
                    'max_bets_per_month': 80,
                    'max_concurrent_bets': 3,
                    'min_time_between_bets': 3600  # 1 hour
                },
                'market_impact': {
                    'enabled': True,
                    'impact_factor': 0.1,
                    'max_impact': 0.05,
                    'recovery_time': 24  # hours
                },
                'diversification': {
                    'enabled': True,
                    'max_bets_per_league': 2,
                    'max_bets_per_team': 1,
                    'max_bets_per_outcome': 1,
                    'correlation_threshold': 0.7
                }
            }
        else:
            self.config = config
        
        # Initialize capital
        self.current_capital = self.config['capital_management']['initial_capital']
        self.peak_capital = self.current_capital
        
    def calculate_kelly_fraction(self, probability: float, odds: float, 
                                recent_performance: List[float] = None) -> float:
        """Calculate Kelly fraction with conservative adjustments"""
        if not self.config['kelly_criterion']['enabled']:
            return 0
        
        if probability <= 0 or probability >= 1 or odds <= 1:
            return 0
        
        # Basic Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_optimal = (b * p - q) / b
        
        # Apply conservative fraction
        kelly_conservative = kelly_optimal * self.config['kelly_criterion']['kelly_fraction']
        
        # Adjust based on recent performance
        if recent_performance:
            recent_accuracy = np.mean(recent_performance[-10:]) if len(recent_performance) >= 10 else 0.5
            
            if recent_accuracy > 0.6:
                # Good performance - increase Kelly slightly
                kelly_conservative *= self.config['kelly_criterion']['kelly_recovery']
            elif recent_accuracy < 0.4:
                # Poor performance - decrease Kelly
                kelly_conservative *= self.config['kelly_criterion']['kelly_decay']
        
        # Apply bounds
        min_kelly = self.config['kelly_criterion']['min_kelly']
        max_kelly = self.config['kelly_criterion']['max_kelly']
        
        kelly_fraction = max(min_kelly, min(max_kelly, kelly_conservative))
        
        return kelly_fraction
    
    def calculate_position_size(self, probability: float, odds: float, 
                              available_capital: float = None) -> float:
        """Calculate position size using Kelly Criterion and risk management"""
        if available_capital is None:
            available_capital = self.current_capital
        
        # Calculate Kelly fraction
        recent_performance = [bet['outcome'] == 'win' for bet in self.betting_history[-20:]]
        kelly_fraction = self.calculate_kelly_fraction(probability, odds, recent_performance)
        
        if kelly_fraction <= 0:
            return 0
        
        # Calculate base position size
        base_position_size = available_capital * kelly_fraction
        
        # Apply risk management adjustments
        position_size = self._apply_risk_management(base_position_size, available_capital)
        
        # Apply diversification constraints
        position_size = self._apply_diversification_constraints(position_size, probability, odds)
        
        # Apply market impact adjustments
        position_size = self._apply_market_impact(position_size, available_capital)
        
        return position_size
    
    def _apply_risk_management(self, position_size: float, available_capital: float) -> float:
        """Apply risk management constraints to position size"""
        # Check drawdown limits
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if current_drawdown > self.config['risk_management']['max_drawdown']:
            self.logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
            return 0
        
        # Reduce position size based on drawdown
        if current_drawdown > 0:
            drawdown_factor = 1 - (current_drawdown / self.config['risk_management']['max_drawdown'])
            position_size *= drawdown_factor
        
        # Check daily loss limit
        today_bets = [bet for bet in self.betting_history 
                     if bet['date'].date() == datetime.now().date()]
        today_loss = sum([bet['net_profit'] for bet in today_bets if bet['net_profit'] < 0])
        
        if abs(today_loss) > self.current_capital * self.config['risk_management']['daily_loss_limit']:
            self.logger.warning("Daily loss limit reached")
            return 0
        
        # Check weekly loss limit
        week_start = datetime.now() - timedelta(days=7)
        week_bets = [bet for bet in self.betting_history if bet['date'] >= week_start]
        week_loss = sum([bet['net_profit'] for bet in week_bets if bet['net_profit'] < 0])
        
        if abs(week_loss) > self.current_capital * self.config['risk_management']['weekly_loss_limit']:
            self.logger.warning("Weekly loss limit reached")
            return 0
        
        # Apply position size limits
        max_position = available_capital * self.config['capital_management']['max_position_size']
        min_position = available_capital * self.config['capital_management']['min_position_size']
        
        position_size = max(min_position, min(max_position, position_size))
        
        return position_size
    
    def _apply_diversification_constraints(self, position_size: float, 
                                        probability: float, odds: float) -> float:
        """Apply diversification constraints to position size"""
        if not self.config['diversification']['enabled']:
            return position_size
        
        # Check concurrent bets limit
        active_bets = [bet for bet in self.betting_history 
                      if bet['status'] == 'active']
        
        if len(active_bets) >= self.config['betting_limits']['max_concurrent_bets']:
            self.logger.warning("Maximum concurrent bets reached")
            return 0
        
        # Check daily bet limit
        today_bets = [bet for bet in self.betting_history 
                     if bet['date'].date() == datetime.now().date()]
        
        if len(today_bets) >= self.config['betting_limits']['max_bets_per_day']:
            self.logger.warning("Daily bet limit reached")
            return 0
        
        # Check weekly bet limit
        week_start = datetime.now() - timedelta(days=7)
        week_bets = [bet for bet in self.betting_history if bet['date'] >= week_start]
        
        if len(week_bets) >= self.config['betting_limits']['max_bets_per_week']:
            self.logger.warning("Weekly bet limit reached")
            return 0
        
        # Check monthly bet limit
        month_start = datetime.now() - timedelta(days=30)
        month_bets = [bet for bet in self.betting_history if bet['date'] >= month_start]
        
        if len(month_bets) >= self.config['betting_limits']['max_bets_per_month']:
            self.logger.warning("Monthly bet limit reached")
            return 0
        
        # Reduce position size based on diversification
        diversification_factor = 1.0
        
        # Reduce if too many bets on same outcome
        same_outcome_bets = [bet for bet in active_bets 
                            if bet['prediction'] == self._get_outcome_from_odds(odds)]
        
        if len(same_outcome_bets) > 0:
            diversification_factor *= 0.5
        
        position_size *= diversification_factor
        
        return position_size
    
    def _apply_market_impact(self, position_size: float, available_capital: float) -> float:
        """Apply market impact adjustments to position size"""
        if not self.config['market_impact']['enabled']:
            return position_size
        
        # Calculate market impact
        impact_factor = self.config['market_impact']['impact_factor']
        max_impact = self.config['market_impact']['max_impact']
        
        # Impact increases with position size relative to capital
        relative_size = position_size / available_capital
        market_impact = min(max_impact, relative_size * impact_factor)
        
        # Reduce position size to account for market impact
        adjusted_position_size = position_size * (1 - market_impact)
        
        return adjusted_position_size
    
    def _get_outcome_from_odds(self, odds: float) -> int:
        """Determine outcome from odds (simplified)"""
        if odds < 2.0:
            return 2  # Home win
        elif odds < 3.0:
            return 0  # Away win
        else:
            return 1  # Draw
    
    def should_place_bet(self, probability: float, odds: float, 
                        confidence: float, match_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Determine if a bet should be placed"""
        decision = {
            'should_bet': False,
            'reason': '',
            'position_size': 0,
            'kelly_fraction': 0,
            'confidence_score': 0
        }
        
        # Check confidence threshold
        if confidence < self.config['confidence_filtering']['min_confidence']:
            decision['reason'] = 'low_confidence'
            return decision
        
        if confidence > self.config['confidence_filtering']['max_confidence']:
            decision['reason'] = 'overconfident'
            return decision
        
        # Check odds range
        if odds < self.config['odds_filtering']['min_odds']:
            decision['reason'] = 'odds_too_low'
            return decision
        
        if odds > self.config['odds_filtering']['max_odds']:
            decision['reason'] = 'odds_too_high'
            return decision
        
        # Calculate position size
        position_size = self.calculate_position_size(probability, odds)
        
        if position_size <= 0:
            decision['reason'] = 'no_position_size'
            return decision
        
        # Check minimum position size
        min_position = self.current_capital * self.config['capital_management']['min_position_size']
        
        if position_size < min_position:
            decision['reason'] = 'position_too_small'
            return decision
        
        # Check if we have enough capital
        if position_size > self.current_capital * 0.8:  # Leave 20% reserve
            decision['reason'] = 'insufficient_capital'
            return decision
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(probability, odds)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(probability, odds, confidence)
        
        # All checks passed
        decision['should_bet'] = True
        decision['reason'] = 'approved'
        decision['position_size'] = position_size
        decision['kelly_fraction'] = kelly_fraction
        decision['confidence_score'] = confidence_score
        
        return decision
    
    def _calculate_confidence_score(self, probability: float, odds: float, 
                                   confidence: float) -> float:
        """Calculate overall confidence score"""
        # Base confidence
        base_score = confidence
        
        # Odds-based adjustment
        odds_weight = self.config['odds_filtering']['odds_weight']
        preferred_range = self.config['odds_filtering']['preferred_odds_range']
        
        if preferred_range[0] <= odds <= preferred_range[1]:
            odds_score = 1.0
        else:
            # Penalize odds outside preferred range
            if odds < preferred_range[0]:
                odds_score = odds / preferred_range[0]
            else:
                odds_score = preferred_range[1] / odds
        
        # Combine scores
        overall_score = base_score * (1 - odds_weight) + odds_score * odds_weight
        
        return overall_score
    
    def place_bet(self, probability: float, odds: float, confidence: float,
                  match_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Place a bet based on the strategy"""
        # Check if bet should be placed
        decision = self.should_place_bet(probability, odds, confidence, match_info)
        
        if not decision['should_bet']:
            return {
                'bet_placed': False,
                'reason': decision['reason'],
                'decision': decision
            }
        
        # Create bet record
        bet = {
            'id': len(self.betting_history) + 1,
            'date': datetime.now(),
            'probability': probability,
            'odds': odds,
            'confidence': confidence,
            'position_size': decision['position_size'],
            'kelly_fraction': decision['kelly_fraction'],
            'confidence_score': decision['confidence_score'],
            'status': 'placed',
            'match_info': match_info or {},
            'net_profit': 0,
            'outcome': None
        }
        
        # Update capital
        self.current_capital -= decision['position_size']
        
        # Record bet
        self.betting_history.append(bet)
        
        self.logger.info(f"Bet placed: ID={bet['id']}, Size=${decision['position_size']:.2f}, "
                        f"Odds={odds:.2f}, Confidence={confidence:.3f}")
        
        return {
            'bet_placed': True,
            'bet': bet,
            'decision': decision
        }
    
    def settle_bet(self, bet_id: int, actual_outcome: int) -> Dict[str, Any]:
        """Settle a bet with the actual outcome"""
        # Find the bet
        bet = None
        for b in self.betting_history:
            if b['id'] == bet_id:
                bet = b
                break
        
        if not bet:
            return {'error': 'Bet not found'}
        
        if bet['status'] != 'placed':
            return {'error': 'Bet already settled'}
        
        # Determine if bet won
        predicted_outcome = self._get_outcome_from_odds(bet['odds'])
        won = predicted_outcome == actual_outcome
        
        # Calculate profit/loss
        if won:
            gross_profit = bet['position_size'] * (bet['odds'] - 1)
            # Apply commission and slippage
            commission = bet['position_size'] * 0.05  # 5% commission
            slippage = bet['position_size'] * 0.02   # 2% slippage
            net_profit = gross_profit - commission - slippage
        else:
            net_profit = -bet['position_size']
        
        # Update bet
        bet['status'] = 'settled'
        bet['actual_outcome'] = actual_outcome
        bet['net_profit'] = net_profit
        bet['outcome'] = 'win' if won else 'loss'
        bet['settlement_date'] = datetime.now()
        
        # Update capital
        self.current_capital += bet['position_size'] + net_profit
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Update current drawdown
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        self.logger.info(f"Bet settled: ID={bet_id}, Outcome={'WIN' if won else 'LOSS'}, "
                        f"Profit=${net_profit:.2f}, Capital=${self.current_capital:.2f}")
        
        return {
            'bet_settled': True,
            'bet': bet,
            'won': won,
            'net_profit': net_profit,
            'current_capital': self.current_capital
        }
    
    def get_betting_stats(self) -> Dict[str, Any]:
        """Get current betting statistics"""
        if not self.betting_history:
            return {}
        
        # Basic stats
        total_bets = len(self.betting_history)
        settled_bets = [bet for bet in self.betting_history if bet['status'] == 'settled']
        active_bets = [bet for bet in self.betting_history if bet['status'] == 'placed']
        
        if settled_bets:
            winning_bets = [bet for bet in settled_bets if bet['outcome'] == 'win']
            losing_bets = [bet for bet in settled_bets if bet['outcome'] == 'loss']
            
            win_rate = len(winning_bets) / len(settled_bets)
            total_profit = sum([bet['net_profit'] for bet in settled_bets])
            avg_profit = total_profit / len(settled_bets)
            
            # Risk metrics
            max_drawdown = self.current_drawdown
            sharpe_ratio = self._calculate_sharpe_ratio(settled_bets)
            
            # Position sizing stats
            avg_position_size = np.mean([bet['position_size'] for bet in settled_bets])
            avg_kelly_fraction = np.mean([bet['kelly_fraction'] for bet in settled_bets])
            avg_confidence = np.mean([bet['confidence'] for bet in settled_bets])
            
            stats = {
                'total_bets': total_bets,
                'settled_bets': len(settled_bets),
                'active_bets': len(active_bets),
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_position_size': avg_position_size,
                'avg_kelly_fraction': avg_kelly_fraction,
                'avg_confidence': avg_confidence
            }
        else:
            stats = {
                'total_bets': total_bets,
                'settled_bets': 0,
                'active_bets': len(active_bets),
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'current_drawdown': self.current_drawdown
            }
        
        self.betting_stats = stats
        return stats
    
    def _calculate_sharpe_ratio(self, settled_bets: List[Dict]) -> float:
        """Calculate Sharpe ratio for settled bets"""
        if len(settled_bets) < 2:
            return 0
        
        returns = [bet['net_profit'] / bet['position_size'] for bet in settled_bets]
        
        if np.std(returns) == 0:
            return 0
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        return sharpe_ratio
    
    def reset_strategy(self):
        """Reset the betting strategy"""
        self.current_capital = self.config['capital_management']['initial_capital']
        self.peak_capital = self.current_capital
        self.current_drawdown = 0
        self.betting_history = []
        self.betting_stats = {}
        
        self.logger.info("Betting strategy reset")
    
    def save_strategy_state(self, filepath: str):
        """Save current strategy state"""
        import joblib
        
        strategy_state = {
            'betting_history': self.betting_history,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': self.current_drawdown,
            'betting_stats': self.betting_stats,
            'config': self.config
        }
        
        joblib.dump(strategy_state, filepath)
        self.logger.info(f"Strategy state saved to {filepath}")
    
    def load_strategy_state(self, filepath: str):
        """Load strategy state"""
        import joblib
        
        strategy_state = joblib.load(filepath)
        
        self.betting_history = strategy_state['betting_history']
        self.current_capital = strategy_state['current_capital']
        self.peak_capital = strategy_state['peak_capital']
        self.current_drawdown = strategy_state['current_drawdown']
        self.betting_stats = strategy_state['betting_stats']
        self.config = strategy_state['config']
        
        self.logger.info(f"Strategy state loaded from {filepath}")

# Example usage
def main():
    """Example usage of NonMajorLeagueBettingStrategy"""
    
    # Initialize betting strategy
    strategy = NonMajorLeagueBettingStrategy()
    
    # Simulate some bets
    test_cases = [
        {'probability': 0.7, 'odds': 2.5, 'confidence': 0.8},
        {'probability': 0.6, 'odds': 3.0, 'confidence': 0.7},
        {'probability': 0.8, 'odds': 1.8, 'confidence': 0.9},
        {'probability': 0.5, 'odds': 4.0, 'confidence': 0.6},
    ]
    
    for i, test_case in enumerate(test_cases):
        # Place bet
        result = strategy.place_bet(
            probability=test_case['probability'],
            odds=test_case['odds'],
            confidence=test_case['confidence']
        )
        
        if result['bet_placed']:
            bet = result['bet']
            print(f"Bet {bet['id']} placed: ${bet['position_size']:.2f} at {bet['odds']:.2f} odds")
            
            # Simulate outcome (random for demo)
            actual_outcome = np.random.choice([0, 1, 2])
            
            # Settle bet
            settlement = strategy.settle_bet(bet['id'], actual_outcome)
            if settlement['bet_settled']:
                print(f"Bet {bet['id']} settled: {'WIN' if settlement['won'] else 'LOSS'}, "
                      f"Profit: ${settlement['net_profit']:.2f}")
        else:
            print(f"Bet rejected: {result['reason']}")
    
    # Get statistics
    stats = strategy.get_betting_stats()
    print(f"\nBetting Statistics:")
    print(f"Total bets: {stats['total_bets']}")
    print(f"Win rate: {stats['win_rate']:.2%}")
    print(f"Total profit: ${stats['total_profit']:.2f}")
    print(f"Current capital: ${stats['current_capital']:.2f}")
    print(f"Current drawdown: {stats['current_drawdown']:.2%}")

if __name__ == "__main__":
    main()
