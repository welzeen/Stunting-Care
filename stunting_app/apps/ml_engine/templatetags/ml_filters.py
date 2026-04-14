from django import template

register = template.Library()


@register.filter
def dict_get(d, key):
    if isinstance(d, dict):
        return d.get(str(key), d.get(key, ''))
    return ''


@register.filter
def index(lst, i):
    try:
        return lst[i]
    except (IndexError, TypeError):
        return ''


@register.filter
def enumerate(lst):
    try:
        return list(builtins_enumerate(lst))
    except Exception:
        return []


# Override with python built-in
import builtins
builtins_enumerate = builtins.enumerate


@register.filter(name='enumerate')
def enumerate_filter(lst):
    try:
        return list(builtins.enumerate(lst))
    except Exception:
        return []


@register.filter
def add_str(value, arg):
    return str(value) + str(arg)


@register.filter
def to_persen(value):
    """Ubah nilai desimal (0.8905) menjadi persen (89.05%)"""
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return "-"


@register.filter
def dict_get_persen(d, key):
    """Ambil nilai dari dict lalu ubah ke persen."""
    try:
        val = d.get(str(key), d.get(key, 0)) if isinstance(d, dict) else 0
        return f"{float(val) * 100:.2f}%"
    except (ValueError, TypeError):
        return "-"


@register.filter(name='get_item')
def get_item(d, key):
    """Akses dict dengan variabel di template: dict|get_item:key"""
    if isinstance(d, dict):
        return d.get(key, '')
    return ''
