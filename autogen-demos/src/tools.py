from typing import Literal, Optional

from .constants import *
from .logging import trace_logger
from .model.base import RouterLLM

g20_with_eu_members = [
    # G20 Countries
    "Argentina",
    "Australia",
    "Brazil",
    "Canada",
    "China",
    "France",  # EU member
    "Germany",  # EU member
    "India",
    "Indonesia",
    "Italy",  # EU member
    "Japan",
    "Mexico",
    "Russia",
    "Saudi Arabia",
    "South Africa",
    "South Korea",
    "Turkey",
    "United Kingdom",
    "United States",
    "European Union",  # Regional member

    # EU Member Countries (27 total)
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Republic of Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",  # Already in G20
    "Germany",  # Already in G20
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",  # Already in G20
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden"
]


def get_policy_context_tool(module: Literal[
    'social_security_or_insurance_or_emiratisation_schemes', 'employment_service_related_queries', 'labor_law_and_employment_policy_query', 'migration_or_visa_or_workforce_mobility_query', 'mohre_or_government_policy_or_administration_query'],
                            countries: Optional[list[str]] = ['UAE']) -> dict:
    """Retrieves the relevant information about Labor Policy Documents (laws, amendments, unemployment insurance, social security, employment schemes, migration policies, visa/workforce mobility policies, etc) pre-stored in our Vector Database.
        Args:
            module (str): Type of user query
            countries (List[str]): The name of the countries for which the policy report needs to be retrieved. Defaults to ['UAE'] if no country was specified.
        Returns:
            dict: status and result or error msg.
    """

    if not countries:
        countries = 'UAE'

    else:
        countries = ', '.join(countries)

    try:
        return {
            "status": "success",
            "report": (
                f"The information is present in Policy documents and can be extracted via module: {module}",
                f"The country/countries referenced here was/were: {countries}"
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"No Information found from Vector Database is not available."
        }


def get_data_context_tool(
        module: Literal['economic_data_query', 'labor_force_or_database_query', 'markdown_data_query']) -> dict:
    """Returns the data context w.r.t to the module workflow chosen for context retrieval.


    Args:
        module (str): Type of user query.

    Returns:
        dict: status and result or error msg.
    """

    try:
        return {
            "status": "success",
            "report": (
                f"The data is present in {module}"
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"No Information found from Vector Database is not available."
        }


def get_general_context_tool(module: Literal[
    'conversational_query', 'general_knowledge_based_queries', 'harmful_or_offensive_query']) -> dict:
    """Returns the selected module context. Handles all kinds of general question but blocks harmful content

    Args:
        module (str): Type of user query.

    Returns:
        dict: status and result or error msg.
    """

    try:
        return {
            "status": "success",
            "report": (
                f"The information is present in {module}"
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"No Information found from Vector Database is not available."
        }


def check_grammar_tool(text_input: str) -> dict:
    """Checks the grammar of input text and returns corrections and explanations.

    This function uses the Gemini API to analyze the provided text for
    grammatical errors. It returns a dictionary containing the corrected
    text, explanations of the errors, and descriptions of the errors.  The
    Gemini API is called with a prompt requesting a JSON response in a
    specific format.  The function handles potential errors in communicating
    with the Gemini API and parsing the JSON response.

    Args:
        text_input: The input text to be checked for grammar errors.

    Returns:
        A dictionary containing the following keys:
            "corrected_text": The text with grammatical errors corrected, or None
                            if an error occurred.
            "explanations": A list of strings explaining each correction made, or
                            a list containing an error message if an error occurred.
            "errors": A list of strings describing each error found.  This may
                      be an empty list if no errors were found or if an error
                      occurred during processing.

        The dictionary structure is designed to conform to the following JSON schema:

        ```json
        {
          "type": "object",
          "properties": {
            "corrected_text": {
              "type": "string",
              "description": "The corrected text."
            },
            "explanations": {
              "type": "array",
              "items": {
                "type": "string",
                "description": "Explanation of a specific error."
              },
              "description": "An array of explanations for each correction."
            },
            "errors": {
              "type": "array",
              "items": {
                "type": "string",
                "description": "Description of a specific error."
              },
              "description": "An array of descriptions of each error."
            }
          },
          "required": [
            "corrected_text",
            "explanations",
            "errors"
          ],
          "description": "A JSON object containing corrected text, explanations, and error descriptions."
        }
        ```

        If an error occurs during communication with the Gemini API or parsing
        the JSON response, the "corrected_text" will be None and the "explanations"
        list will contain an error message. The "errors" list might be empty
        in such cases.
    """

    prompt = f"""
    Analyze the following text for grammar errors, correct them, and provide 
    explanations for each correction:

    Text: {text_input}

    Return the response as a JSON object with the following structure:
    {{
      "corrected_text": "The corrected text.",
      "explanations": [
        "Explanation of the first error.",
        "Explanation of the second error.",
        ...
      ],
      "errors": [
          "Description of the first error",
          "Description of the second error",
          ...
      ]
    }}
    """

    from typing import List
    from pydantic import BaseModel, Field

    class ResponseSchema(BaseModel):
        """
        A JSON object containing corrected text, explanations, and error descriptions.
        """
        corrected_text: str = Field(
            ...,
            description="The corrected text."
        )
        explanations: List[str] = Field(
            ...,
            description="An array of explanations for each correction."
        )
        errors: List[str] = Field(
            ...,
            description="An array of descriptions of each error."
        )

    try:

        llm = RouterLLM(
            model=LLAMA3_2_API_MODEL_NAME,
            openai_api_base=LLAMA3_2_API_BASE,
            openai_api_key=LLAMA3_2_API_KEY,
            temperature=0.4,
            top_p=0.1
        )

        response: ResponseSchema = llm.structured_output_to_pydantic_model(prompt, response_model=ResponseSchema)

        try:
            trace_logger.info(f"Response Recorded Here: {response}")
            return response.__dict__
        except Exception as e:
            return {
                "corrected_text": None,
                "explanations": [f"Error parsing LLAMA3.2 3B response: {e}"],
                "errors": []
            }

    except Exception as e:
        return {
            "corrected_text": None,
            "explanations": [f"Error communicating with LLAMA3.2 3B: {e}"],
            "errors": []
        }
