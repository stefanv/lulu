"""
Editor factory that overrides certain attributes of the default editor.

For example, the default editor for Range(low=0, high=1500) is a spin box.
To change it to a slider instead, use

my_range = Range(low=0, high=1500, editor=DefaultOverride(mode='slider'))

Alternatively, the editor can also be specified in the view:

View(Item('my_range', editor=DefaultOverride(mode='slider'))

"""

from enthought.traits.api import Dict
from enthought.traits.ui.editor_factory import EditorFactory

factory_traits = EditorFactory().trait_names()

class DefaultOverride(EditorFactory):
    """Editor factory for selectively overriding certain parameters
    of the default editor.

    """
    _overrides = Dict

    def __init__(self, *args, **overrides):
        factory_kwds = {}
        for k in overrides:
            if k in factory_traits:
                factory_kwds[k] = overrides.pop(k)

        EditorFactory.__init__(self, *args, **factory_kwds)
        self._overrides = overrides

    def _customise_default(self, editor_kind, ui, object, name,
                           description, parent):
        """
        Obtain the given trait's default editor and set the parameters
        specified in `overrides` above.
        """
        trait = getattr(object, 'trait')(name)
        editor_factory = trait.trait_type.create_editor()
        for option in self._overrides:
            setattr(editor_factory, option, self._overrides[option])

        editor = getattr(editor_factory, editor_kind)(ui, object, name,
                                                      description, parent)
        return editor

    def simple_editor(self, *args):
        return self._customise_default('simple_editor', *args)

    def custom_editor(self, *args):
        return self._customise_default('custom_editor', *args)

    def text_editor(self, *args):
        return self._customise_default('text_editor', *args)

    def readonly_editor(self, *args):
        return self._customise_default('readonly_editor', *args)
