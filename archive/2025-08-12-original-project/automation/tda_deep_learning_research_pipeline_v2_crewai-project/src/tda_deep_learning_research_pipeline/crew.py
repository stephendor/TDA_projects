import os
from dotenv import load_dotenv
load_dotenv("../../.env")

import os
from dotenv import load_dotenv

# Force load environment variables BEFORE any CrewAI imports
load_dotenv('../../.env')

# Ensure the API key is set in the environment
if not os.getenv('GOOGLE_API_KEY'):
    raise Exception("GOOGLE_API_KEY not found in environment!")

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
# Import tools directly to bypass problematic __init__.py
try:
    from crewai_tools.tools.serply_search_tools.serply_scholar_search_tool import SerplyScholarSearchTool
    from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
    print("✅ Successfully imported tools directly")
except ImportError as e:
    print(f"⚠️ Direct import failed: {e}")
    # Try the main import as fallback
    try:
        from crewai_tools import ScrapeWebsiteTool
        # Create a basic search tool if scholar search fails
        class SerplyScholarSearchTool:
            def __init__(self):
                self.name = "Basic Scholar Search"
        print("✅ Using fallback imports")
    except ImportError:
        print("❌ All imports failed - creating dummy tools")
        class SerplyScholarSearchTool:
            def __init__(self):
                self.name = "Dummy Scholar Search"
        class ScrapeWebsiteTool:
            def __init__(self):
                self.name = "Dummy Web Scraper"



@CrewBase
class TdaDeepLearningResearchPipelineCrew:
    """TdaDeepLearningResearchPipeline crew"""

    
    @agent
    def tda_research_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["tda_research_specialist"],
            # tools=[],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gemini/gemini-2.5-pro",
                temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY"),
            ),
        )
    
    @agent
    def tda_analysis_engineer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["tda_analysis_engineer"],
            # tools=[],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gemini/gemini-2.5-pro",
                temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY"),
            ),
        )
    
    @agent
    def deep_learning_engineer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["deep_learning_engineer"],
            # tools=[],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gemini/gemini-2.5-pro",
                temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY"),
            ),
        )
    
    @agent
    def technical_report_writer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["technical_report_writer"],
            tools=[],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gemini/gemini-2.5-pro",
                temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY"),
            ),
        )
    

    
    @task
    def literature_review_on_tda_applications(self) -> Task:
        return Task(
            config=self.tasks_config["literature_review_on_tda_applications"],
        )
    
    @task
    def tda_feature_extraction_and_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["tda_feature_extraction_and_analysis"],
        )
    
    @task
    def deep_learning_model_development(self) -> Task:
        return Task(
            config=self.tasks_config["deep_learning_model_development"],
        )
    
    @task
    def comprehensive_technical_report(self) -> Task:
        return Task(
            config=self.tasks_config["comprehensive_technical_report"],
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the TdaDeepLearningResearchPipeline crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
