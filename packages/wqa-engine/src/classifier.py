"""
Page type classification logic.

Classifies URLs into an expanded taxonomy based on URL path patterns.
"""

import pandas as pd
from urllib.parse import urlparse


def classify_page_type(url: str, page_title: str = '', content_type: str = '') -> str:
    """
    Classify page into expanded taxonomy:
    - Home Page: Root/homepage
    - Local Lander: Location/city/area pages
    - Landing Page: Service/product/solution pages
    - Lead Generation: Contact, quote, demo request pages
    - Blog Post: Individual blog articles
    - Blog Category: Blog listing/category/tag pages
    - Resource / Guide: Resources, guides, whitepapers, case studies
    - Site Info: About, privacy, terms, careers pages
    - Other: Anything else
    """
    if not url or pd.isna(url):
        return 'Other'
    url_lower = str(url).lower()
    title_lower = str(page_title).lower() if page_title else ''

    try:
        parsed = urlparse(url_lower)
        path = parsed.path.replace('\\', '/')
    except Exception:
        path = url_lower.replace('\\', '/')

    # Home Page - root path only
    if path in ['/', ''] or path.rstrip('/') == '':
        return 'Home Page'

    # Lead Generation - contact, quote, demo pages (check early - high priority)
    lead_gen_patterns = ['/contact', '/get-quote', '/request-quote', '/free-quote',
                         '/demo', '/request-demo', '/schedule', '/book-', '/appointment',
                         '/consultation', '/free-estimate', '/get-started', '/signup', '/sign-up',
                         '/register', '/inquiry', '/enquiry']
    if any(pattern in path for pattern in lead_gen_patterns):
        return 'Lead Generation'

    # Blog Category - blog listing/category/tag pages (check before Blog Post)
    blog_category_patterns = ['/blog/', '/blog', '/news/', '/news', '/articles/', '/posts/']
    # Check if it's a category/tag/author page or the main blog listing
    if any(pattern in path for pattern in blog_category_patterns):
        # If path ends with /blog or /blog/ or has category/tag/author/page patterns
        category_indicators = ['/category/', '/tag/', '/author/', '/page/', '/topics/']
        if (path.rstrip('/').endswith('/blog') or
            path.rstrip('/').endswith('/news') or
            path.rstrip('/').endswith('/articles') or
            any(ind in path for ind in category_indicators)):
            return 'Blog Category'
        # Otherwise it's likely a blog post (has additional path segments)
        return 'Blog Post'

    # Local Lander - location/city/area pages
    local_patterns = ['/location/', '/locations/', '/city/', '/cities/',
                      '/service-area/', '/service-areas/', '/area/', '/areas/',
                      '/near-me', '/local/', '/region/', '/state/', '/county/']
    if any(pattern in path for pattern in local_patterns):
        return 'Local Lander'

    # Resource / Guide - educational content, downloads, case studies
    resource_patterns = ['/resource/', '/resources/', '/guide/', '/guides/',
                        '/whitepaper/', '/whitepapers/', '/ebook/', '/ebooks/',
                        '/case-study/', '/case-studies/', '/casestudy/', '/casestudies/',
                        '/download/', '/downloads/', '/library/', '/learn/',
                        '/how-to/', '/tutorial/', '/tutorials/', '/faq/', '/faqs/',
                        '/knowledge-base/', '/kb/', '/help-center/', '/documentation/']
    if any(pattern in path for pattern in resource_patterns):
        return 'Resource / Guide'

    # Site Info - corporate/legal/info pages
    site_info_patterns = ['/about', '/privacy', '/terms', '/legal/', '/disclaimer',
                          '/careers/', '/jobs/', '/team/', '/leadership/',
                          '/cookie', '/gdpr', '/accessibility', '/sitemap',
                          '/press/', '/media/', '/awards/', '/testimonials/',
                          '/partners/', '/affiliate/', '/referral/']
    if any(pattern in path for pattern in site_info_patterns):
        return 'Site Info'

    # Landing Page - service/product/solution pages
    landing_patterns = ['/service/', '/services/', '/solutions/', '/solution/',
                       '/products/', '/product/', '/what-we-do/', '/offerings/',
                       '/industries/', '/industry/', '/capabilities/', '/expertise/']
    if any(pattern in path for pattern in landing_patterns):
        return 'Landing Page'

    return 'Other'
