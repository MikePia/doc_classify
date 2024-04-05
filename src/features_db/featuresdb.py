import csv
import json
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text


# Define the base class
Base = declarative_base()


engine = None


def get_engine(force=False, connection_string="sqlite:///features.db"):
    """
    Returns a SQLAlchemy engine instance. If it doesn't exist or if 'force' is True, it creates a new engine instance.

    Parameters:
    - force (bool): If True, force the creation of a new engine instance even if one already exists.
    - connection_string (str): The database connection string.

    Returns:
    - engine: A SQLAlchemy engine instance.
    """
    global engine
    if not engine or force:
        engine = create_engine(connection_string)
    return engine


Session = sessionmaker(bind=get_engine())


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    classification_id = Column(Integer, ForeignKey("classifications.id"))
    classification = relationship("Classification", back_populates="documents")
    features = relationship(
        "Feature", secondary="document_features", back_populates="documents"
    )


class Classification(Base):
    __tablename__ = "classifications"
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)
    documents = relationship("Document", back_populates="classification")


class Feature(Base):
    __tablename__ = "features"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    documents = relationship(
        "Document", secondary="document_features", back_populates="features"
    )


# Redefined DocumentFeature class to include relationship description
class DocumentFeature(Base):
    __tablename__ = "document_features"
    document_id = Column(Integer, ForeignKey("documents.id"), primary_key=True)
    feature_id = Column(Integer, ForeignKey("features.id"), primary_key=True)
    description = Column(
        Text, nullable=True
    )  # Additional field to describe the feature's relation to the document

    # Relationships to enable easy access back to the Document and Feature
    document = relationship(Document)
    feature = relationship(Feature)



def createdb():

    # Create an engine that stores data in the local directory's features.db file.
    engine = create_engine("sqlite:///features.db")

    # Create all tables in the engine. This is equivalent to "Create Table" statements in raw SQL.
    Base.metadata.create_all(engine)
    return engine


def populate_db(engine):

    # Instantiate a session
    session = Session()

    classification1 = Classification(type="Presentation")
    classification2 = Classification(type="Not Presentation")
    classification3 = Classification(type="Mix")

    # feature1 = Feature(name="Contains Slides")
    # feature2 = Feature(name="Text Heavy")
    # feature3 = Feature(name="Mixed Content")

    # document1 = Document(
    #     name="Annual Report", classification=classification2, features=[feature2]
    # )
    # document2 = Document(
    #     name="Company Overview",
    #     classification=classification1,
    #     features=[feature1, feature3],
    # )

    # Add example data to the session and commit to the database
    session.add_all(
        [
            classification1,
            classification2,
            classification3,
            # feature1,
            # feature2,
            # feature3,
            # document1,
            # document2,
        ]
    )
    session.commit()

    # Close the session
    session.close()


def find_document_by_name(document_name):
    """
    Finds and returns a document from the 'documents' table by its name, managing the session internally.

    Parameters:
    - document_name: The name of the document to find.

    Returns:
    - The Document object if found, otherwise None.
    """
    # Open a new session
    session = Session()

    try:
        # Query the database for a document with the specified name
        document = (
            session.query(Document).filter(Document.name == document_name).first()
        )
        # Return the document if found, else None
        return document
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Ensure the session is closed
        session.close()


def add_document_with_features(document_name, classification_type, features_with_descriptions):
    """
    Adds a document with associated features and descriptions to the database.

    Parameters:
    - document_name: The name of the document.
    - classification_type: The classification type of the document.
    - features_with_descriptions: A dictionary with feature names as keys and their descriptions
      as values in relation to the document.

    Returns:
    - None
    """

    session = Session()
    # Check if the classification exists; if not, create a new one
    classification = session.query(Classification).filter_by(type=classification_type).first()
    if not classification:
        classification = Classification(type=classification_type)
        session.add(classification)
        session.commit()

    # Create a new Document instance
    new_document = Document(name=document_name, classification=classification)
    session.add(new_document)

    for feature_name, description in features_with_descriptions.items():
        # Check if the feature exists; if not, create a new one
        feature = session.query(Feature).filter_by(name=feature_name).first()
        if not feature:
            feature = Feature(name=feature_name)
            session.add(feature)
            session.commit()  # Ensure the feature is committed to obtain its ID

        # Create a new DocumentFeature instance
        description_json = json.dumps(description)
        document_feature = DocumentFeature(document=new_document, feature=feature, description=description_json)
        session.add(document_feature)

    # Commit the new document, its features, and the relationships to the database
    session.commit()


def get_features_from_type(classifier, filter=None):
    session = Session()
    presentations = session.query(Document).filter_by(classification_id=classifier).all()
    features = []
    for presentation in presentations:
        for feature in presentation.features:
            if filter:
                if filter not in feature.name.lower():
                    continue
                else:
                    # Get documentfeature description
                    docfeat = session.query(DocumentFeature).filter_by(document_id=presentation.id, feature_id=feature.id).first()

                    features.append((presentation.id, feature.name, docfeat.description))

            else:
                features.append((presentation.id, feature.name))
    return features

if __name__ == "__main__":
    # engine = createdb()
    # populate_db(engine)
    # print("Database and tables created with example data.")

    # # Example usage
    # document_name = "Annual Report"
    # found_document = find_document_by_name(document_name)

    # if found_document:
    #     print(f"Document Found: ID = {found_document.id}, Name = {found_document.name}")
    # else:
    #     print("Document not found.")
    features = get_features_from_type(1, filter="invest")
    # Save to csv
    with open('data/analysis/invest.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['presentation_id', 'feature_name'])
        for feature in features:
            writer.writerow(feature)
    print()