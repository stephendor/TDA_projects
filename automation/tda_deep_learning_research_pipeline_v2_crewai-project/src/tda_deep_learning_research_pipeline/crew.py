import os
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	SerplyScholarSearchTool,
	ScrapeWebsiteTool
)



@CrewBase
class TdaDeepLearningResearchPipelineCrew:
    """TdaDeepLearningResearchPipeline crew"""

    
    @agent
    def tda_research_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["tda_research_specialist"],
            tools=[SerplyScholarSearchTool()],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
        )
    
    @agent
    def tda_analysis_engineer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["tda_analysis_engineer"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
        )
    
    @agent
    def deep_learning_engineer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["deep_learning_engineer"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
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
                model="gpt-4o-mini",
                temperature=0.7,
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
