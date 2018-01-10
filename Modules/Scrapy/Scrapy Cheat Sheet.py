from scrapy.selector import Selector
from scrapy.http import HtmlResponse



	# Example: 
	HTML = """
	<div class="postinginfos">
	    <p class="postinginfo">post id: 5400585892</p>
	    <p class="postinginfo">posted: <time datetime="2016-01-12T23:23:19-0800" class="xh-highlight">2016-01-12 11:23pm</time></p>
	    <p class="postinginfo"><a href="https://accounts.craigslist.org/eaf?postingID=5400585892" class="tsb">email to friend</a></p>
	    <p class="postinginfo"><a class="bestof-link" data-flag="9" href="https://post.craigslist.org/flag?flagCode=9&amp;postingID=5400585892" title="nominate for best-of-CL"><span class="bestof-icon">♥ </span><span class="bestof-text">best of</span></a> <sup>[<a href="http://www.craigslist.org/about/best-of-craigslist">?</a>]</sup>    </p>
	</div>
	"""

	best = Selector(text=HTML).xpath("//span[@class='bestof-text']/text()").extract()