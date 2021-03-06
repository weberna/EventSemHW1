#Contains the SPARQL queries for the 5 defined semantic roles
#Each are constructed to pick out edges which meet the requisete generalized role definition



##AGENT Edge Queries
agent_querystr ="""
                SELECT ?edge
                WHERE { ?pred ?edge ?arg;
                              <domain> <semantics>;
                              <type> <predicate> .
                        ?arg <domain> <semantics>;
                             <type> <argument> .
                        ?edge <existed_before> ?existed_before FILTER ( ?existed_before > 0 ) . 
                        { ?edge <volition> ?volition FILTER ( ?volition > 0 ) } UNION { ?edge <instigation> ?instigation FILTER ( ?instigation > 0 ) } .
                      }
                """

##PATIENT Edge Query
#Patient undergoes some type of change and does not instigate the event
patient_querystr ="""
                    SELECT ?edge
                    WHERE { ?pred ?edge ?arg;
                                  <domain> <semantics>;
                                  <type> <predicate> .
                            ?arg <domain> <semantics>;
                                 <type> <argument> .
                            ?edge <existed_before> ?existed_before FILTER ( ?existed_before > 0 ) . 
                            ?edge <instigation> ?instigation FILTER ( ?instigation <= 0 ) . 
                            { ?edge <change_of_state> ?cos FILTER ( ?cos > 0 ) } UNION 
                            { ?edge <change_of_location> ?col FILTER ( ?col > 0 ) } UNION 
                            { ?edge <change_of_possession> ?cop FILTER ( ?cop > 0 ) } .
                          }
                """

##INSTRAMENT Edge Query
#Instrament is used and does not instigate the event
inst_querystr ="""
                    SELECT ?edge
                    WHERE { ?pred ?edge ?arg;
                                  <domain> <semantics>;
                                  <type> <predicate> .
                            ?arg <domain> <semantics>;
                                 <type> <argument> .
                            ?edge <existed_before> ?existed_before FILTER ( ?existed_before > 0 ) . 
                            ?edge <instigation> ?instigation FILTER ( ?instigation <= 0 ) . 
                            ?edge <was_used> ?used FILTER ( ?used > 0 ) .
                          }
                """
#BENIFICIARY Edge Query
#Benificiaries do not change state, location, or possestion, have the event done to their benefit, and do not instigate the event
benificiary_querystr ="""
                    SELECT ?edge
                    WHERE { ?pred ?edge ?arg;
                                  <domain> <semantics>;
                                  <type> <predicate> .
                            ?arg <domain> <semantics>;
                                 <type> <argument> .
                            ?edge <existed_before> ?existed_before FILTER ( ?existed_before > 0 ) . 
                            ?edge <instigation> ?instigation FILTER ( ?instigation <= 0 ) . 
                            ?edge <was_for_benefit> ?benefit FILTER ( ?benefit > 0 ) .
                            ?edge <change_of_state> ?cos FILTER ( ?cos <= 0 ) .
                            ?edge <change_of_location> ?col FILTER ( ?col <= 0 ) .
                            ?edge <change_of_possession> ?cop FILTER ( ?cop <= 0 )  .
                          }
                """
#EXPIERENCER Edge Query 
#Expierencers have sentience and awareness of the event, but do not instigate it
exp_querystr ="""
                    SELECT ?edge
                    WHERE { ?pred ?edge ?arg;
                                  <domain> <semantics>;
                                  <type> <predicate> .
                            ?arg <domain> <semantics>;
                                 <type> <argument> .
                            ?edge <existed_before> ?existed_before FILTER ( ?existed_before > 0 ) . 
                            ?edge <instigation> ?instigation FILTER ( ?instigation <= 0 ) . 
                            ?edge <awareness> ?awareness FILTER ( ?awareness > 0 ) .
                            ?edge <sentient> ?sentient FILTER ( ?sentient > 0 ) .
                          }
                """

