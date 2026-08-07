"""Human-readable class names used for CLIP text prompts and diffusion prompts.

The order of every list matches the *original* label order of the corresponding
torchvision dataset (i.e. before any FSCIL class-order permutation is applied):

* CIFAR100_CLASSES  -- alphabetical order of torchvision.datasets.CIFAR100.
* CUB200_CLASSES    -- alphabetical order of the CUB-200-2011 folder names.
* IMAGENETR_CLASSES -- alphabetical order of the 200 ImageNet-R WordNet ids,
  which is exactly the order torchvision.datasets.ImageFolder produces.

models.asp.Learner._get_class_names re-indexes these lists with
args["class_order"] so that they line up with the mapped training labels.
"""

import os
import warnings

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
    "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle",
    "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest",
    "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower",
    "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
    "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

CUB200_CLASSES = [
    "Black footed Albatross", "Laysan Albatross", "Sooty Albatross", "Groove billed Ani",
    "Crested Auklet", "Least Auklet", "Parakeet Auklet", "Rhinoceros Auklet",
    "Brewer Blackbird", "Red winged Blackbird", "Rusty Blackbird", "Yellow headed Blackbird",
    "Bobolink", "Indigo Bunting", "Lazuli Bunting", "Painted Bunting", "Cardinal",
    "Spotted Catbird", "Gray Catbird", "Yellow breasted Chat", "Eastern Towhee",
    "Chuck will Widow", "Brandt Cormorant", "Red faced Cormorant", "Pelagic Cormorant",
    "Bronzed Cowbird", "Shiny Cowbird", "Brown Creeper", "American Crow", "Fish Crow",
    "Black billed Cuckoo", "Mangrove Cuckoo", "Yellow billed Cuckoo",
    "Gray crowned Rosy Finch", "Purple Finch", "Northern Flicker", "Acadian Flycatcher",
    "Great Crested Flycatcher", "Least Flycatcher", "Olive sided Flycatcher",
    "Scissor tailed Flycatcher", "Vermilion Flycatcher", "Yellow bellied Flycatcher",
    "Frigatebird", "Northern Fulmar", "Gadwall", "American Goldfinch", "European Goldfinch",
    "Boat tailed Grackle", "Eared Grebe", "Horned Grebe", "Pied billed Grebe",
    "Western Grebe", "Blue Grosbeak", "Evening Grosbeak", "Pine Grosbeak",
    "Rose breasted Grosbeak", "Pigeon Guillemot", "California Gull", "Glaucous winged Gull",
    "Heermann Gull", "Herring Gull", "Ivory Gull", "Ring billed Gull", "Slaty backed Gull",
    "Western Gull", "Anna Hummingbird", "Ruby throated Hummingbird", "Rufous Hummingbird",
    "Green Violetear", "Long tailed Jaeger", "Pomarine Jaeger", "Blue Jay", "Florida Jay",
    "Green Jay", "Dark eyed Junco", "Tropical Kingbird", "Gray Kingbird", "Belted Kingfisher",
    "Green Kingfisher", "Pied Kingfisher", "Ringed Kingfisher", "White breasted Kingfisher",
    "Red legged Kittiwake", "Horned Lark", "Pacific Loon", "Mallard", "Western Meadowlark",
    "Hooded Merganser", "Red breasted Merganser", "Mockingbird", "Nighthawk",
    "Clark Nutcracker", "White breasted Nuthatch", "Baltimore Oriole", "Hooded Oriole",
    "Orchard Oriole", "Scott Oriole", "Ovenbird", "Brown Pelican", "White Pelican",
    "Western Wood Pewee", "Sayornis", "American Pipit", "Whip poor Will", "Horned Puffin",
    "Common Raven", "White necked Raven", "American Redstart", "Geococcyx",
    "Loggerhead Shrike", "Great Grey Shrike", "Baird Sparrow", "Black throated Sparrow",
    "Brewer Sparrow", "Chipping Sparrow", "Clay colored Sparrow", "House Sparrow",
    "Field Sparrow", "Fox Sparrow", "Grasshopper Sparrow", "Harris Sparrow",
    "Henslow Sparrow", "Le Conte Sparrow", "Lincoln Sparrow", "Nelson Sharp tailed Sparrow",
    "Savannah Sparrow", "Seaside Sparrow", "Song Sparrow", "Tree Sparrow", "Vesper Sparrow",
    "White crowned Sparrow", "White throated Sparrow", "Cape Glossy Starling", "Bank Swallow",
    "Barn Swallow", "Cliff Swallow", "Tree Swallow", "Scarlet Tanager", "Summer Tanager",
    "Artic Tern", "Black Tern", "Caspian Tern", "Common Tern", "Elegant Tern",
    "Forsters Tern", "Least Tern", "Green tailed Towhee", "Brown Thrasher", "Sage Thrasher",
    "Black capped Vireo", "Blue headed Vireo", "Philadelphia Vireo", "Red eyed Vireo",
    "Warbling Vireo", "White eyed Vireo", "Yellow throated Vireo", "Bay breasted Warbler",
    "Black and white Warbler", "Black throated Blue Warbler", "Blue winged Warbler",
    "Canada Warbler", "Cape May Warbler", "Cerulean Warbler", "Chestnut sided Warbler",
    "Golden winged Warbler", "Hooded Warbler", "Kentucky Warbler", "Magnolia Warbler",
    "Mourning Warbler", "Myrtle Warbler", "Nashville Warbler", "Orange crowned Warbler",
    "Palm Warbler", "Pine Warbler", "Prairie Warbler", "Prothonotary Warbler",
    "Swainson Warbler", "Tennessee Warbler", "Wilson Warbler", "Worm eating Warbler",
    "Yellow Warbler", "Northern Waterthrush", "Louisiana Waterthrush", "Bohemian Waxwing",
    "Cedar Waxwing", "American Three toed Woodpecker", "Pileated Woodpecker",
    "Red bellied Woodpecker", "Red cockaded Woodpecker", "Red headed Woodpecker",
    "Downy Woodpecker", "Bewick Wren", "Cactus Wren", "Carolina Wren", "House Wren",
    "Marsh Wren", "Rock Wren", "Winter Wren", "Common Yellowthroat",
]

IMAGENETR_CLASSES = [
    "goldfish", "great white shark", "hammerhead shark", "stingray", "hen", "ostrich",
    "goldfinch", "junco", "bald eagle", "vulture", "smooth newt", "axolotl", "tree frog",
    "green iguana", "chameleon", "Indian cobra", "scorpion", "tarantula", "centipede",
    "peacock", "lorikeet", "hummingbird", "toucan", "duck", "goose", "black swan", "koala",
    "jellyfish", "snail", "American lobster", "hermit crab", "flamingo", "great egret",
    "pelican", "king penguin", "grey whale", "killer whale", "sea lion", "Chihuahua",
    "Shih Tzu", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Italian Greyhound",
    "Whippet", "Weimaraner", "Yorkshire Terrier", "Boston Terrier", "Scottish Terrier",
    "West Highland White Terrier", "Golden Retriever", "Labrador Retriever", "Cocker Spaniels",
    "collie", "Border Collie", "Rottweiler", "German Shepherd Dog", "Boxer", "French Bulldog",
    "St. Bernard", "Siberian Husky", "Dalmatian", "pug", "Pomeranian", "Chow Chow",
    "Pembroke Welsh Corgi", "Toy Poodle", "Standard Poodle", "grey wolf", "hyena", "red fox",
    "tabby cat", "leopard", "snow leopard", "lion", "tiger", "cheetah", "polar bear", "meerkat",
    "ladybug", "fly", "bee", "ant", "grasshopper", "cockroach", "mantis", "dragonfly",
    "monarch butterfly", "starfish", "cottontail rabbit", "porcupine", "fox squirrel", "beaver",
    "guinea pig", "zebra", "pig", "hippopotamus", "bison", "gazelle", "llama", "skunk",
    "badger", "orangutan", "gorilla", "chimpanzee", "gibbon", "baboon", "giant panda", "eel",
    "clownfish", "pufferfish", "accordion", "ambulance", "assault rifle", "backpack", "barn",
    "wheelbarrow", "basketball", "bathtub", "lighthouse", "beer glass", "binoculars",
    "birdhouse", "bow tie", "broom", "bucket", "cauldron", "candle", "cannon", "canoe",
    "carousel", "castle", "mobile phone", "cowboy hat", "electric guitar", "fire engine",
    "flute", "gas mask", "grand piano", "guillotine", "hammer", "harmonica", "harp", "hatchet",
    "jeep", "joystick", "lab coat", "lawn mower", "lipstick", "mailbox", "missile", "mitten",
    "parachute", "pickup truck", "pirate ship", "revolver", "rugby ball", "sandal", "saxophone",
    "school bus", "schooner", "shield", "soccer ball", "space shuttle", "spider web",
    "steam locomotive", "scarf", "submarine", "tank", "tennis ball", "tractor", "trombone",
    "vase", "violin", "military aircraft", "wine bottle", "ice cream", "bagel", "pretzel",
    "cheeseburger", "hot dog", "cabbage", "broccoli", "cucumber", "bell pepper", "mushroom",
    "Granny Smith", "strawberry", "lemon", "pineapple", "banana", "pomegranate", "pizza",
    "burrito", "espresso", "volcano", "baseball player", "scuba diver", "acorn",
]

CLASS_NAMES = {
    "cifar100": CIFAR100_CLASSES,
    "cifar224": CIFAR100_CLASSES,
    "cub": CUB200_CLASSES,
    "cub200": CUB200_CLASSES,
    "imagenetr": IMAGENETR_CLASSES,
}


def get_class_names(dataset_name):
    """Return the ordered class-name list for dataset_name.

    Raises a helpful error instead of silently falling back, because a wrong
    name list would silently corrupt every CLIP score and diffusion prompt.
    """
    key = str(dataset_name).lower()
    if key not in CLASS_NAMES:
        raise ValueError(
            "No text class names registered for dataset '{}'. "
            "Add them to utils/class_names.py before enabling CLIP curation "
            "or diffusion synthesis.".format(dataset_name)
        )
    return CLASS_NAMES[key]


def load_imagenetr_classname_map(classnames_txt):
    """Load ``<wnid> <readable name>`` lines from an optional text file."""
    if not classnames_txt or not os.path.isfile(classnames_txt):
        return {}

    mapping = {}
    with open(classnames_txt, encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(":", " ", 1).split(maxsplit=1)
            if len(parts) != 2 or not parts[0].startswith("n"):
                warnings.warn(
                    "Ignoring malformed ImageNet-R class-name line {}: {!r}".format(
                        line_number, line
                    )
                )
                continue
            mapping[parts[0]] = parts[1].strip().replace("_", " ")
    return mapping


def resolve_imagenetr_class_names(wnids, classnames_txt=None):
    """Resolve readable names in the exact ``ImageFolder.classes`` order.

    The bundled 200-name list preserves the original implementation when a
    standard full ImageNet-R tree is used.  ``classnames.txt`` is preferred
    because it makes the WNID/name relationship explicit and also supports
    custom mirrors or subsets without relying on positional assumptions.
    """
    wnids = [str(x) for x in wnids]
    explicit = load_imagenetr_classname_map(classnames_txt)

    if explicit:
        missing = [wnid for wnid in wnids if wnid not in explicit]
        if missing:
            warnings.warn(
                "{} ImageNet-R folders are missing from {}. Falling back for: {}".format(
                    len(missing), classnames_txt, missing[:10]
                )
            )
        positional = IMAGENETR_CLASSES if len(wnids) == len(IMAGENETR_CLASSES) else None
        return [
            explicit.get(wnid, positional[i] if positional is not None else wnid)
            for i, wnid in enumerate(wnids)
        ]

    if len(wnids) == len(IMAGENETR_CLASSES):
        return list(IMAGENETR_CLASSES)

    warnings.warn(
        "ImageNet-R has {} class folders instead of 200 and no classnames.txt; "
        "using WNIDs as prompt text.".format(len(wnids))
    )
    return wnids
