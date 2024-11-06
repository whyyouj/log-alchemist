# Implemented this else there will be circular imports

from typing import TypedDict, Annotated, List, Union, Optional, Dict, Any
from langchain_core.agents import AgentAction, AgentFinish
import operator
import pandas as pd

class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    pandas: Any
    df: pd.DataFrame
    remaining_qns: list
    all_answer: list 
    parsed_query: Optional[Dict]