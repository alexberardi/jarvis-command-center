import re
from typing import List, Dict, Set, Tuple, Any
from app.request_models.voice_command_request import CommandDefinition
from rapidfuzz import fuzz, process
import random

class CommandFilter:
    def __init__(self):
        pass
    
    def extract_relevant_commands(self, voice_command: str, available_commands: List[CommandDefinition]) -> Tuple[List[CommandDefinition], Dict[str, Any]]:
        """
        Extract commands that are likely relevant to the voice command based on keywords.
        Uses fuzzy matching with multi-tier confidence strategy:
        - >95%: Only high confidence matches
        - 85-95%: All matches in this range
        - 70-85%: Progressive inclusion with fallbacks
        - <70%: All commands (no filtering)
        """
        if not available_commands:
            return available_commands, {"total_commands": 0, "filtered_commands": 0, "reduction_percentage": 0}
        
        voice_lower = voice_command.lower()
        
        # Start with ALL commands as the default
        relevant_commands = available_commands.copy()
        matched_commands = set()
        
        # Check for multiple commands in the voice command
        conjunction_words = ['and', 'then', 'also', 'plus', 'as well as', 'after that']
        has_multiple_commands = any(word in voice_lower for word in conjunction_words)
        
        # Pass 1: Look for strong keyword matches (exact substring)
        strong_matches = []
        for cmd in available_commands:
            if cmd.keywords:
                for keyword in cmd.keywords:
                    if keyword.lower() in voice_lower:
                        if cmd.command_name not in matched_commands:
                            strong_matches.append(cmd)
                            matched_commands.add(cmd.command_name)
                        break
        
        # Pass 2: Look for fuzzy keyword matches
        fuzzy_matches = []
        fuzzy_threshold = 70  # Lower threshold to catch more potential matches
        
        for cmd in available_commands:
            if cmd.command_name not in matched_commands and cmd.keywords:
                best_match_score = 0
                best_match_keyword = None
                
                for keyword in cmd.keywords:
                    # Use rapidfuzz for fuzzy matching
                    scores = [
                        fuzz.partial_ratio(keyword.lower(), voice_lower),  # Best for partial matches
                        fuzz.ratio(keyword.lower(), voice_lower),          # Overall similarity
                        fuzz.token_sort_ratio(keyword.lower(), voice_lower),  # Word order independent
                        fuzz.token_set_ratio(keyword.lower(), voice_lower)     # Word set similarity
                    ]
                    
                    max_score = max(scores)
                    if max_score > best_match_score:
                        best_match_score = max_score
                        best_match_keyword = keyword
                
                if best_match_score >= fuzzy_threshold:
                    fuzzy_matches.append((cmd, best_match_score, best_match_keyword))
                    matched_commands.add(cmd.command_name)
        
        # Sort fuzzy matches by score (highest first)
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Apply multi-tier confidence strategy
        if strong_matches:
            # Strong matches found - use them as primary
            relevant_commands = strong_matches.copy()
            
            # Add fuzzy matches based on confidence tiers
            for cmd, score, keyword in fuzzy_matches:
                if score >= 85:  # High confidence - always include
                    if cmd not in relevant_commands:
                        relevant_commands.append(cmd)
                elif score >= 70:  # Medium confidence - add with fallbacks
                    if cmd not in relevant_commands:
                        relevant_commands.append(cmd)
        
        elif fuzzy_matches:
            # No strong matches, but fuzzy matches found
            max_score = fuzzy_matches[0][1] if fuzzy_matches else 0
            
            if max_score > 95:
                # Very high confidence - only include high confidence matches
                high_confidence_matches = [cmd for cmd, score, keyword in fuzzy_matches if score > 95]
                relevant_commands = high_confidence_matches.copy()
                
            elif max_score >= 85:
                # High confidence - include all matches in this range
                high_confidence_matches = [cmd for cmd, score, keyword in fuzzy_matches if score >= 85]
                relevant_commands = high_confidence_matches.copy()
                
                # Add 1 fallback for context
                fallback_candidates = [cmd for cmd in available_commands if cmd not in relevant_commands]
                if fallback_candidates:
                    relevant_commands.append(random.choice(fallback_candidates))
                    
            elif max_score >= 80:
                # Medium-high confidence - include all matches + fallbacks
                medium_matches = [cmd for cmd, score, keyword in fuzzy_matches if score >= 80]
                relevant_commands = medium_matches.copy()
                
                # Add 2 fallbacks
                fallback_candidates = [cmd for cmd in available_commands if cmd not in relevant_commands]
                if fallback_candidates:
                    fallbacks = random.sample(fallback_candidates, min(2, len(fallback_candidates)))
                    relevant_commands.extend(fallbacks)
                    
            elif max_score >= 75:
                # Medium confidence - top 4 matches + fallbacks
                top_matches = [cmd for cmd, score, keyword in fuzzy_matches[:4]]
                relevant_commands = top_matches.copy()
                
                # Add 2 fallbacks
                fallback_candidates = [cmd for cmd in available_commands if cmd not in relevant_commands]
                if fallback_candidates:
                    fallbacks = random.sample(fallback_candidates, min(2, len(fallback_candidates)))
                    relevant_commands.extend(fallbacks)
                    
            elif max_score >= 70:
                # Low-medium confidence - top 3 matches + fallbacks
                top_matches = [cmd for cmd, score, keyword in fuzzy_matches[:3]]
                relevant_commands = top_matches.copy()
                
                # Add 3 fallbacks
                fallback_candidates = [cmd for cmd in available_commands if cmd not in relevant_commands]
                if fallback_candidates:
                    fallbacks = random.sample(fallback_candidates, min(3, len(fallback_candidates)))
                    relevant_commands.extend(fallbacks)
            else:
                # Below 70% - keep all commands (no filtering)
                pass
        
        # Pass 3: Look for regex pattern matches (word boundaries) as backup
        regex_matches = []
        for cmd in available_commands:
            if cmd.command_name not in matched_commands and cmd.keywords:
                for keyword in cmd.keywords:
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    if re.search(pattern, voice_lower):
                        regex_matches.append(cmd)
                        matched_commands.add(cmd.command_name)
                        break
        
        # Add regex matches if we don't have enough commands
        if len(relevant_commands) < 3 and regex_matches:
            for cmd in regex_matches:
                if cmd not in relevant_commands:
                    relevant_commands.append(cmd)
        
        # If no keyword matches found, keep all commands (no filtering)
        # This is the correct behavior - we only filter when we have confidence
        
        # Calculate statistics
        total_commands = len(available_commands)
        filtered_commands = len(relevant_commands)
        reduction_percentage = ((total_commands - filtered_commands) / total_commands * 100) if total_commands > 0 else 0
        
        # Estimate token savings (rough approximation)
        avg_tokens_per_command = 15  # Rough estimate
        tokens_saved = (total_commands - filtered_commands) * avg_tokens_per_command
        
        # Collect fuzzy match details for logging
        fuzzy_match_details = []
        for cmd, score, keyword in fuzzy_matches:
            fuzzy_match_details.append({
                "command": cmd.command_name,
                "keyword": keyword,
                "score": score
            })
        
        # Determine confidence tier for logging
        max_fuzzy_score = max([score for _, score, _ in fuzzy_matches]) if fuzzy_matches else 0
        confidence_tier = "unknown"
        if max_fuzzy_score > 95:
            confidence_tier = "very_high"
        elif max_fuzzy_score >= 85:
            confidence_tier = "high"
        elif max_fuzzy_score >= 80:
            confidence_tier = "medium_high"
        elif max_fuzzy_score >= 75:
            confidence_tier = "medium"
        elif max_fuzzy_score >= 70:
            confidence_tier = "low_medium"
        else:
            confidence_tier = "low"
        
        stats = {
            "total_commands": total_commands,
            "filtered_commands": filtered_commands,
            "reduction_percentage": reduction_percentage,
            "tokens_saved": tokens_saved,
            "has_multiple_commands": has_multiple_commands,
            "strong_matches": len(strong_matches),
            "fuzzy_matches": len(fuzzy_matches),
            "regex_matches": len(regex_matches),
            "fuzzy_match_details": fuzzy_match_details,
            "max_fuzzy_score": max_fuzzy_score,
            "confidence_tier": confidence_tier
        }
        
        return relevant_commands, stats 