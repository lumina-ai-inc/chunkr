from chunkr_ai.api.chunkr_async import ChunkrAsync
from chunkr_ai.api.config import Configuration, JsonSchema
from chunkr_ai.api.schema import from_pydantic
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from pprint import pprint
import json
from enum import Enum

chunkr = ChunkrAsync()

class ContactMethod(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    BOTH = "both"

class ContactInfo(BaseModel):
    phone: str = Field(description="Contact phone number")
    email: str = Field(description="Contact email address")
    preferred_method: ContactMethod = Field(description="Preferred contact method")

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    postal_code: str = Field(description="Postal/ZIP code")
    contact: ContactInfo = Field(description="Contact information for this address")

class Industry(str, Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    OTHER = "other"

class Company(BaseModel):
    name: str = Field(description="Company name")
    industry: Industry = Field(description="Company industry")
    address: Address = Field(description="Company headquarters address")

class Department(BaseModel):
    name: str = Field(description="Department name")
    head_count: int = Field(description="Number of employees in department")
    location: Address = Field(description="Department location")

class TestModel(BaseModel):
    name: str = Field(description="The user's full name")
    age: int = Field(description="User's age in years", gt=0)
    email: Optional[str] = Field(None, description="User's email address")
    tags: List[str] = Field(default_factory=list, description="List of tags associated with the user")
    home_address: Address = Field(description="User's home address")
    employer: Company = Field(description="User's employer information")
    department: Department = Field(description="User's department information")

async def main():
    # Test with class
    class_schema = from_pydantic(TestModel)
    print("Schema from class:")
    pprint(class_schema)
    
    # Test with instance
    contact_info = ContactInfo(
        phone="123-456-7890",
        email="contact@example.com",
        preferred_method=ContactMethod.EMAIL
    )
    
    address = Address(
        street="123 Main St",
        city="Example City",
        country="Example Country",
        postal_code="12345",
        contact=contact_info
    )
    
    company = Company(
        name="Example Corp",
        industry=Industry.TECHNOLOGY,
        address=address
    )
    
    department = Department(
        name="Engineering",
        head_count=50,
        location=address
    )
    
    test_instance = TestModel(
        name="Test User",
        age=25,
        email="test@example.com",
        tags=["test", "example"],
        home_address=address,
        employer=company,
        department=department
    )
    
    instance_schema = from_pydantic(test_instance)
    print("\nSchema from instance:")
    pprint(instance_schema)
    
class CampusEnrollment(BaseModel):
    undergraduate: int = Field(description="Number of undergraduate students")
    graduate: int = Field(description="Number of graduate students")

class TermEnrollment(BaseModel):
    term: str = Field(description="Academic term (e.g., 'Fall 2019')")
    eugene: CampusEnrollment = Field(description="Enrollment numbers for Eugene campus")
    portland: CampusEnrollment = Field(description="Enrollment numbers for Portland campus")

class EnrollmentReport(BaseModel):
    title: str = Field(description="Report title")
    description: str = Field(description="Report description")
    last_updated: datetime = Field(description="Last update timestamp")
    enrollments: List[TermEnrollment] = Field(description="List of enrollment data by term")

async def test_enrollment_schema():
    # Create test data
    report = EnrollmentReport(
        title="University of Oregon Enrollment Report",
        description="Enrollment statistics by campus and student level",
        last_updated=datetime.now(),
        enrollments=[
            TermEnrollment(
                term="Fall 2019",
                eugene=CampusEnrollment(
                    undergraduate=19886,
                    graduate=3441
                ),
                portland=CampusEnrollment(
                    undergraduate=1024,
                    graduate=208
                )
            ),
            TermEnrollment(
                term="Winter 2020",
                eugene=CampusEnrollment(
                    undergraduate=19660,
                    graduate=3499
                ),
                portland=CampusEnrollment(
                    undergraduate=1026,
                    graduate=200
                )
            ),
            TermEnrollment(
                term="Spring 2020",
                eugene=CampusEnrollment(
                    undergraduate=19593,
                    graduate=3520
                ),
                portland=CampusEnrollment(
                    undergraduate=998,
                    graduate=211
                )
            )
        ]
    )

    # Test schema generation
    class_schema = from_pydantic(EnrollmentReport)
    print("\nSchema from EnrollmentReport class:")
    pprint(class_schema)
    
    instance_schema = from_pydantic(report)
    print("\nSchema from EnrollmentReport instance:")
    pprint(instance_schema)
    
    # Upload to Chunkr
    task = await chunkr.upload("../../tests/files/test.pdf", Configuration(
        json_schema=instance_schema
    ))
    await task.poll_async()

if __name__ == "__main__":
    asyncio.run(test_enrollment_schema())

if __name__ == "__main__":
    # asyncio.run(main())
    # openai_schema()
    asyncio.run(test_enrollment_schema())
