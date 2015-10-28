import plasTeX.Base as Base

def idgen():
    """ Generate a unique ID """
    i = 1
    while 1:
        yield 'a%.10d' % i
        i += 1

idgen = idgen()

class Eqn(Base.Command):
    args = 'self'

class Anchor(Base.Command):
    args = 'label:str'
    def invoke(self, tex):
        Base.Command.invoke(self, tex)
        self.ownerDocument.context.label(self.attributes['label'], self)

class exercise(Base.Environment):
    counter = 'exercise'

class index(Base.Command):
    args = 'termstring'

    def setEntry(self, s, seetype=0):
    #    TYPE_NORMAL = 0
    #    TYPE_SEE = 1
    #    TYPE_SEEALSO = 2
        if type(s) != type(''):
            s = s.textContent
        if s.count('!'):
            priterm, secterm = s.split('!')
            if priterm.count('@'):
                prisort, primary = priterm.split('@')
            else:
                prisort, primary = None, priterm
            if secterm.count('@'):
                secsort, secondary = secterm.split('@')
            else:
                secsort, secondary = None, secterm
        elif s.count('@'):
            prisort, primary = s.split('@')
            secsort, secondary = None, None
        else:
            prisort, primary = None, s
            secsort, secondary = None, None

#        if secondary:
#            self.ownerDocument.userdata.setdefault('index', []).append(\
#                Base.IndexEntry([primary, secondary], self, [prisort, secsort], None, type=seetype))
#        else:
#            self.ownerDocument.userdata.setdefault('index', []).append(\
#                Base.IndexEntry([primary], self, [prisort], None, type=seetype))
        return prisort, primary, secsort, secondary

    def invoke(self, tex):
        Base.Command.invoke(self, tex)
        self.ownerDocument.context.label(idgen.next(), self)
        p0,p1,s0,s1 = self.setEntry(self.attributes['termstring'])
        if p0:
            self.prisort = '%s' % p0
        if p1:
            self.primary = '%s' % p1
        if s0:
            self.secsort = '%s' % s0
        if s1:
            self.secondary = '%s' % s1

class scriptN(Base.Command):
       unicode = u'\U0001D4A9'

class uxbar(Base.Command): pass
class uybar(Base.Command): pass
class unhat(Base.Command): pass
class ule(Base.Command): pass
class minus(Base.Command): pass
class lowast(Base.Command): pass
class Erdos(Base.Command): pass

