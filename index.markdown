---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
Hello World!

<ul class="post-list">
    {% for post in site.posts %}
      <li>
        <a href="{{ post.url }}">{{ post.title }}</a><br>
        {{ post.excerpt | strip_html | truncatewords:75 }}
      </li>
    {% endfor %}
  </ul>