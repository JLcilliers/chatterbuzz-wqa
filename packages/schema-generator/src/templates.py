"""
Phase 7 — JSON-LD Schema Templates

Nine page-type template functions that each accept a data dict and return
a fully-formed JSON-LD dictionary ready for serialisation.
"""

import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _base(schema_type: str) -> dict:
    """Return the minimal JSON-LD envelope."""
    return {
        "@context": "https://schema.org",
        "@type": schema_type,
    }


def _get(data: dict, key: str, default: Any = None) -> Any:
    """Safe dict access with optional default."""
    return data.get(key, default)


# ---------------------------------------------------------------------------
# Template functions
# ---------------------------------------------------------------------------

def organization_schema(data: dict) -> dict:
    """Organization schema — company / brand identity."""
    schema = _base("Organization")
    schema.update({
        "name": _get(data, "name", ""),
        "url": _get(data, "url", ""),
        "logo": _get(data, "logo", ""),
        "description": _get(data, "description", ""),
        "sameAs": _get(data, "social_profiles", []),
        "contactPoint": {
            "@type": "ContactPoint",
            "telephone": _get(data, "phone", ""),
            "contactType": _get(data, "contact_type", "customer service"),
            "areaServed": _get(data, "area_served", "US"),
            "availableLanguage": _get(data, "languages", ["English"]),
        },
        "address": {
            "@type": "PostalAddress",
            "streetAddress": _get(data, "street_address", ""),
            "addressLocality": _get(data, "city", ""),
            "addressRegion": _get(data, "state", ""),
            "postalCode": _get(data, "postal_code", ""),
            "addressCountry": _get(data, "country", "US"),
        },
    })
    if _get(data, "founding_date"):
        schema["foundingDate"] = data["founding_date"]
    if _get(data, "founders"):
        schema["founder"] = [
            {"@type": "Person", "name": name} for name in data["founders"]
        ]
    return schema


def article_schema(data: dict) -> dict:
    """Article / BlogPosting schema."""
    article_type = _get(data, "article_type", "Article")
    if article_type not in ("Article", "BlogPosting", "NewsArticle", "TechArticle"):
        article_type = "Article"

    schema = _base(article_type)
    schema.update({
        "headline": _get(data, "headline", ""),
        "description": _get(data, "description", ""),
        "url": _get(data, "url", ""),
        "image": _get(data, "image", ""),
        "datePublished": _get(data, "date_published", ""),
        "dateModified": _get(data, "date_modified", ""),
        "author": {
            "@type": _get(data, "author_type", "Person"),
            "name": _get(data, "author_name", ""),
        },
        "publisher": {
            "@type": "Organization",
            "name": _get(data, "publisher_name", ""),
            "logo": {
                "@type": "ImageObject",
                "url": _get(data, "publisher_logo", ""),
            },
        },
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": _get(data, "url", ""),
        },
    })
    if _get(data, "word_count"):
        schema["wordCount"] = data["word_count"]
    if _get(data, "keywords"):
        schema["keywords"] = data["keywords"]
    return schema


def local_business_schema(data: dict) -> dict:
    """LocalBusiness schema — brick-and-mortar / service-area business."""
    business_type = _get(data, "business_type", "LocalBusiness")
    schema = _base(business_type)
    schema.update({
        "name": _get(data, "name", ""),
        "url": _get(data, "url", ""),
        "image": _get(data, "image", ""),
        "telephone": _get(data, "phone", ""),
        "priceRange": _get(data, "price_range", ""),
        "description": _get(data, "description", ""),
        "address": {
            "@type": "PostalAddress",
            "streetAddress": _get(data, "street_address", ""),
            "addressLocality": _get(data, "city", ""),
            "addressRegion": _get(data, "state", ""),
            "postalCode": _get(data, "postal_code", ""),
            "addressCountry": _get(data, "country", "US"),
        },
        "geo": {
            "@type": "GeoCoordinates",
            "latitude": _get(data, "latitude", ""),
            "longitude": _get(data, "longitude", ""),
        },
        "sameAs": _get(data, "social_profiles", []),
    })
    if _get(data, "opening_hours"):
        schema["openingHoursSpecification"] = [
            {
                "@type": "OpeningHoursSpecification",
                "dayOfWeek": oh.get("days", []),
                "opens": oh.get("opens", ""),
                "closes": oh.get("closes", ""),
            }
            for oh in data["opening_hours"]
        ]
    if _get(data, "aggregate_rating"):
        schema["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": data["aggregate_rating"].get("value", ""),
            "reviewCount": data["aggregate_rating"].get("count", ""),
        }
    return schema


def service_schema(data: dict) -> dict:
    """Service schema — professional or business service."""
    schema = _base("Service")
    schema.update({
        "name": _get(data, "name", ""),
        "description": _get(data, "description", ""),
        "url": _get(data, "url", ""),
        "serviceType": _get(data, "service_type", ""),
        "provider": {
            "@type": "Organization",
            "name": _get(data, "provider_name", ""),
            "url": _get(data, "provider_url", ""),
        },
        "areaServed": _get(data, "area_served", ""),
    })
    if _get(data, "offers"):
        schema["offers"] = {
            "@type": "Offer",
            "price": _get(data["offers"], "price", ""),
            "priceCurrency": _get(data["offers"], "currency", "USD"),
            "availability": _get(data["offers"], "availability",
                                  "https://schema.org/InStock"),
        }
    if _get(data, "has_offer_catalog"):
        schema["hasOfferCatalog"] = {
            "@type": "OfferCatalog",
            "name": data["has_offer_catalog"].get("name", ""),
            "itemListElement": [
                {
                    "@type": "Offer",
                    "itemOffered": {
                        "@type": "Service",
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                    },
                }
                for item in data["has_offer_catalog"].get("items", [])
            ],
        }
    return schema


def faq_schema(data: dict) -> dict:
    """FAQPage schema — list of question/answer pairs."""
    schema = _base("FAQPage")
    questions = _get(data, "questions", [])
    schema["mainEntity"] = [
        {
            "@type": "Question",
            "name": q.get("question", ""),
            "acceptedAnswer": {
                "@type": "Answer",
                "text": q.get("answer", ""),
            },
        }
        for q in questions
    ]
    if _get(data, "url"):
        schema["url"] = data["url"]
    return schema


def product_schema(data: dict) -> dict:
    """Product schema — e-commerce or catalog product."""
    schema = _base("Product")
    schema.update({
        "name": _get(data, "name", ""),
        "description": _get(data, "description", ""),
        "image": _get(data, "image", ""),
        "url": _get(data, "url", ""),
        "sku": _get(data, "sku", ""),
        "brand": {
            "@type": "Brand",
            "name": _get(data, "brand", ""),
        },
        "offers": {
            "@type": "Offer",
            "price": _get(data, "price", ""),
            "priceCurrency": _get(data, "currency", "USD"),
            "availability": _get(data, "availability",
                                  "https://schema.org/InStock"),
            "url": _get(data, "url", ""),
            "priceValidUntil": _get(data, "price_valid_until", ""),
        },
    })
    if _get(data, "aggregate_rating"):
        schema["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": data["aggregate_rating"].get("value", ""),
            "bestRating": data["aggregate_rating"].get("best", "5"),
            "reviewCount": data["aggregate_rating"].get("count", ""),
        }
    if _get(data, "reviews"):
        schema["review"] = [
            {
                "@type": "Review",
                "author": {"@type": "Person", "name": r.get("author", "")},
                "reviewRating": {
                    "@type": "Rating",
                    "ratingValue": r.get("rating", ""),
                    "bestRating": r.get("best", "5"),
                },
                "reviewBody": r.get("body", ""),
            }
            for r in data["reviews"]
        ]
    if _get(data, "gtin"):
        schema["gtin"] = data["gtin"]
    if _get(data, "mpn"):
        schema["mpn"] = data["mpn"]
    return schema


def breadcrumb_schema(data: dict) -> dict:
    """BreadcrumbList schema — navigation trail."""
    schema = _base("BreadcrumbList")
    items = _get(data, "items", [])
    schema["itemListElement"] = [
        {
            "@type": "ListItem",
            "position": idx + 1,
            "name": item.get("name", ""),
            "item": item.get("url", ""),
        }
        for idx, item in enumerate(items)
    ]
    return schema


def webpage_schema(data: dict) -> dict:
    """WebPage schema — generic page-level metadata."""
    page_type = _get(data, "page_type", "WebPage")
    if page_type not in ("WebPage", "AboutPage", "ContactPage",
                         "CollectionPage", "ItemPage", "SearchResultsPage"):
        page_type = "WebPage"

    schema = _base(page_type)
    schema.update({
        "name": _get(data, "name", ""),
        "description": _get(data, "description", ""),
        "url": _get(data, "url", ""),
        "inLanguage": _get(data, "language", "en"),
        "isPartOf": {
            "@type": "WebSite",
            "name": _get(data, "site_name", ""),
            "url": _get(data, "site_url", ""),
        },
    })
    if _get(data, "date_published"):
        schema["datePublished"] = data["date_published"]
    if _get(data, "date_modified"):
        schema["dateModified"] = data["date_modified"]
    if _get(data, "primary_image"):
        schema["primaryImageOfPage"] = {
            "@type": "ImageObject",
            "url": data["primary_image"],
        }
    if _get(data, "breadcrumb"):
        schema["breadcrumb"] = breadcrumb_schema(data["breadcrumb"])
    return schema


def how_to_schema(data: dict) -> dict:
    """HowTo schema — step-by-step instructions."""
    schema = _base("HowTo")
    schema.update({
        "name": _get(data, "name", ""),
        "description": _get(data, "description", ""),
        "url": _get(data, "url", ""),
        "image": _get(data, "image", ""),
        "totalTime": _get(data, "total_time", ""),
    })
    if _get(data, "estimated_cost"):
        schema["estimatedCost"] = {
            "@type": "MonetaryAmount",
            "currency": data["estimated_cost"].get("currency", "USD"),
            "value": data["estimated_cost"].get("value", ""),
        }
    if _get(data, "supply"):
        schema["supply"] = [
            {"@type": "HowToSupply", "name": s} for s in data["supply"]
        ]
    if _get(data, "tool"):
        schema["tool"] = [
            {"@type": "HowToTool", "name": t} for t in data["tool"]
        ]
    steps = _get(data, "steps", [])
    schema["step"] = [
        {
            "@type": "HowToStep",
            "name": step.get("name", ""),
            "text": step.get("text", ""),
            "url": step.get("url", ""),
            "image": step.get("image", ""),
        }
        for step in steps
    ]
    return schema


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

SCHEMA_TEMPLATES: Dict[str, Callable[[dict], dict]] = {
    "organization": organization_schema,
    "article": article_schema,
    "blog": article_schema,            # alias
    "local_business": local_business_schema,
    "service": service_schema,
    "faq": faq_schema,
    "product": product_schema,
    "breadcrumb": breadcrumb_schema,
    "webpage": webpage_schema,
    "how_to": how_to_schema,
}
