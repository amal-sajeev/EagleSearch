from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


#creating a transformer encoder object
encoder  = SentenceTransformer("all-MiniLM-L12-v2")
documents = [
    {
        "name": "Taxi Driver",
        "description": "A mentally unstable veteran works as a nighttime taxi driver in New York City.",
        "director": "Martin Scorsese",
        "year": 1976,
        "genre": ["Drama", "Crime", "Psychological Thriller"],
        "actors": ["Robert De Niro", "Jodie Foster", "Cybill Shepherd", "Harvey Keitel"],
        "mpaa_rating": "R",
        "imdb_rating": 8.2
    },
    {
        "name": "Die Hard",
        "description": "An off-duty cop battles terrorists in a Los Angeles skyscraper.",
        "director": "John McTiernan",
        "year": 1988,
        "genre": ["Action", "Thriller"],
        "actors": ["Bruce Willis", "Alan Rickman", "Bonnie Bedelia", "Reginald VelJohnson"],
        "mpaa_rating": "R",
        "imdb_rating": 8.2
    },
    {
        "name": "The Godfather",
        "description": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
        "director": "Francis Ford Coppola",
        "year": 1972,
        "genre": ["Crime", "Drama"],
        "actors": ["Marlon Brando", "Al Pacino", "James Caan", "Diane Keaton"],
        "mpaa_rating": "R",
        "imdb_rating": 9.2
    },
    {
        "name": "Raging Bull",
        "description": "A biographical story of boxer Jake LaMotta's rise and fall.",
        "director": "Martin Scorsese",
        "year": 1980,
        "genre": ["Drama", "Biography", "Sport"],
        "actors": ["Robert De Niro", "Joe Pesci", "Cathy Moriarty"],
        "mpaa_rating": "R",
        "imdb_rating": 8.2
    },
    {
        "name": "Predator",
        "description": "An elite special forces team faces an alien hunter in a Central American jungle.",
        "director": "John McTiernan",
        "year": 1987,
        "genre": ["Action", "Sci-Fi", "Horror"],
        "actors": ["Arnold Schwarzenegger", "Carl Weathers", "Jesse Ventura", "Bill Duke"],
        "mpaa_rating": "R",
        "imdb_rating": 7.8
    },
    {
        "name": "Apocalypse Now",
        "description": "A military captain travels upriver during the Vietnam War to find a renegade colonel.",
        "director": "Francis Ford Coppola",
        "year": 1979,
        "genre": ["Drama", "War"],
        "actors": ["Martin Sheen", "Marlon Brando", "Robert Duvall", "Dennis Hopper"],
        "mpaa_rating": "R",
        "imdb_rating": 8.4
    },
    {
        "name": "Goodfellas",
        "description": "The story of a young man's rise through the ranks of a crime family.",
        "director": "Martin Scorsese",
        "year": 1990,
        "genre": ["Crime", "Drama", "Biography"],
        "actors": ["Ray Liotta", "Robert De Niro", "Joe Pesci", "Lorraine Bracco"],
        "mpaa_rating": "R",
        "imdb_rating": 8.7
    },
    {
        "name": "The Hunt for Red October",
        "description": "A CIA analyst must prove the intentions of a Soviet submarine captain during the Cold War.",
        "director": "John McTiernan",
        "year": 1990,
        "genre": ["Action", "Thriller"],
        "actors": ["Sean Connery", "Alec Baldwin", "Scott Glenn", "Sam Neill"],
        "mpaa_rating": "PG",
        "imdb_rating": 7.6
    },
    {
        "name": "The Conversation",
        "description": "A surveillance expert becomes increasingly paranoid about his latest assignment.",
        "director": "Francis Ford Coppola",
        "year": 1974,
        "genre": ["Drama", "Mystery", "Thriller"],
        "actors": ["Gene Hackman", "John Cazale", "Allen Garfield"],
        "mpaa_rating": "PG",
        "imdb_rating": 7.8
    },
    {
        "name": "Casino",
        "description": "A tale of greed, deception, and power in Las Vegas casino culture.",
        "director": "Martin Scorsese",
        "year": 1995,
        "genre": ["Crime", "Drama"],
        "actors": ["Robert De Niro", "Sharon Stone", "Joe Pesci", "James Woods"],
        "mpaa_rating": "R",
        "imdb_rating": 8.2
    },
    {
        "name": "The Last Action Hero",
        "description": "A young movie fan is magically transported into the fictional world of his favorite action film character.",
        "director": "John McTiernan",
        "year": 1993,
        "genre": ["Action", "Comedy", "Fantasy"],
        "actors": ["Arnold Schwarzenegger", "Austin O'Brien", "Charles Dance"],
        "mpaa_rating": "PG-13",
        "imdb_rating": 6.4
    },
    {
        "name": "The Godfather Part II",
        "description": "The parallel stories of a young Vito Corleone and his son Michael as they expand their criminal empire.",
        "director": "Francis Ford Coppola",
        "year": 1974,
        "genre": ["Crime", "Drama"],
        "actors": ["Al Pacino", "Robert De Niro", "Robert Duvall", "Diane Keaton"],
        "mpaa_rating": "R",
        "imdb_rating": 9.0
    },
    {
        "name": "The Departed",
        "description": "An undercover cop and a mole in the police force attempt to identify each other.",
        "director": "Martin Scorsese",
        "year": 2006,
        "genre": ["Crime", "Drama", "Thriller"],
        "actors": ["Leonardo DiCaprio", "Matt Damon", "Jack Nicholson", "Mark Wahlberg"],
        "mpaa_rating": "R",
        "imdb_rating": 8.5
    },
    {
        "name": "Basic",
        "description": "A DEA agent investigates the disappearance of a legendary Army Ranger drill sergeant.",
        "director": "John McTiernan",
        "year": 2003,
        "genre": ["Action", "Mystery", "Thriller"],
        "actors": ["John Travolta", "Samuel L. Jackson", "Connie Nielsen"],
        "mpaa_rating": "R",
        "imdb_rating": 6.5
    },
    {
        "name": "Dracula",
        "description": "The legendary vampire story based on Bram Stoker's novel.",
        "director": "Francis Ford Coppola",
        "year": 1992,
        "genre": ["Horror", "Romance", "Fantasy"],
        "actors": ["Gary Oldman", "Winona Ryder", "Anthony Hopkins", "Keanu Reeves"],
        "mpaa_rating": "R",
        "imdb_rating": 7.4
    },
    {
        "name": "The Wolf of Wall Street",
        "description": "The true story of stockbroker Jordan Belfort's rise and fall.",
        "director": "Martin Scorsese",
        "year": 2013,
        "genre": ["Biography", "Crime", "Comedy"],
        "actors": ["Leonardo DiCaprio", "Jonah Hill", "Margot Robbie", "Matthew McConaughey"],
        "mpaa_rating": "R",
        "imdb_rating": 8.2
    },
    {
        "name": "Thomas Crown Affair",
        "description": "A wealthy art thief gets caught in a game of cat and mouse with a female insurance investigator.",
        "director": "John McTiernan",
        "year": 1999,
        "genre": ["Crime", "Romance", "Thriller"],
        "actors": ["Pierce Brosnan", "Rene Russo", "Denis Leary"],
        "mpaa_rating": "R",
        "imdb_rating": 6.8
    },
    {
        "name": "The Outsiders",
        "description": "A story of rivalry between two teenage gangs in 1960s Oklahoma.",
        "director": "Francis Ford Coppola",
        "year": 1983,
        "genre": ["Crime", "Drama"],
        "actors": ["C. Thomas Howell", "Matt Dillon", "Ralph Macchio", "Patrick Swayze"],
        "mpaa_rating": "PG",
        "imdb_rating": 7.1
    },
    {
        "name": "Scarface",
        "description": "A Cuban refugee rises to power as a drug kingpin in 1980s Miami through brutal violence, including infamous scenes with machine guns, explosive shootouts, and chainsaw violence. The film climaxes with a massive assault on a mansion featuring heavy weapons and explosions.",
        "director": "Brian De Palma",
        "year": 1983,
        "genre": ["Crime", "Drama"],
        "actors": ["Al Pacino", "Michelle Pfeiffer", "Steven Bauer", "Mary Elizabeth Mastrantonio"],
        "mpaa_rating": "R",
        "imdb_rating": 8.3
    },
    {
        "name": "Carrie",
        "description": "A shy teenage girl with telekinetic powers faces abuse from her peers and religious mother.",
        "director": "Brian De Palma",
        "year": 1976,
        "genre": ["Horror", "Drama"],
        "actors": ["Sissy Spacek", "Piper Laurie", "Amy Irving", "John Travolta"],
        "mpaa_rating": "R",
        "imdb_rating": 7.4
    },
    {
        "name": "The Untouchables",
        "description": "Federal Agent Eliot Ness sets out to take down Al Capone in Prohibition-era Chicago.",
        "director": "Brian De Palma",
        "year": 1987,
        "genre": ["Crime", "Drama", "Thriller"],
        "actors": ["Kevin Costner", "Sean Connery", "Robert De Niro", "Andy Garcia"],
        "mpaa_rating": "R",
        "imdb_rating": 7.9
    },
    {
        "name": "Mission: Impossible",
        "description": "An American agent goes on the run after being falsely accused of being a mole.",
        "director": "Brian De Palma",
        "year": 1996,
        "genre": ["Action", "Adventure", "Thriller"],
        "actors": ["Tom Cruise", "Jon Voight", "Emmanuelle Béart", "Jean Reno"],
        "mpaa_rating": "PG-13",
        "imdb_rating": 7.1
    },
    {
        "name": "Dressed to Kill",
        "description": "A mysterious blonde woman murders a psychiatrist's patient, leading to an investigation.",
        "director": "Brian De Palma",
        "year": 1980,
        "genre": ["Mystery", "Thriller", "Horror"],
        "actors": ["Michael Caine", "Angie Dickinson", "Nancy Allen"],
        "mpaa_rating": "R",
        "imdb_rating": 7.1
    },
    {
        "name": "Body Double",
        "description": "An actor house-sitting becomes obsessed with spying on a neighbor and witnesses a murder.",
        "director": "Brian De Palma",
        "year": 1984,
        "genre": ["Mystery", "Thriller"],
        "actors": ["Craig Wasson", "Melanie Griffith", "Gregg Henry"],
        "mpaa_rating": "R",
        "imdb_rating": 6.8
    },
    {
        "name": "Carlito's Way",
        "description": "A former drug dealer tries to go straight but his past and former connections plague him.",
        "director": "Brian De Palma",
        "year": 1993,
        "genre": ["Crime", "Drama", "Thriller"],
        "actors": ["Al Pacino", "Sean Penn", "Penelope Ann Miller", "John Leguizamo"],
        "mpaa_rating": "R",
        "imdb_rating": 7.9
    },
    {
        "name": "Snake Eyes",
        "description": "A corrupt police officer investigates an assassination at a boxing match.",
        "director": "Brian De Palma",
        "year": 1998,
        "genre": ["Crime", "Mystery", "Thriller"],
        "actors": ["Nicolas Cage", "Gary Sinise", "Carla Gugino"],
        "mpaa_rating": "R",
        "imdb_rating": 6.0
    }
]

client = QdrantClient(":memory:"                                                                            
)

client.create_collection(
    collection_name = "cool_movies",
    vectors_config = models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(),
        distance = models.Distance.COSINE
    )
)

client.upload_points(
    collection_name = "cool_movies",
    points=[
        models.PointStruct(
            id=idx,
            vector = encoder.encode(doc["description"]).tolist(),
            payload = doc
        )
        for idx, doc in enumerate(documents)
    ]
)

hits = client.query_points(
    collection_name="cool_movies",
    query=encoder.encode("Machine guns").tolist(),
    query_filter=models.Filter(
        must = [models.FieldCondition(key = "year",
        range = models.Range(gte = 2000))]
    ),
    limit=5
).points

for hit in hits:
    print(hit.payload,"score", hit.score)